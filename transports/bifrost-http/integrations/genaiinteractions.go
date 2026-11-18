package integrations

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/providers/gemini"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
	"github.com/valyala/fasthttp"
)

// The GenAI Interactions routes serve Google's Interactions API (POST /interactions and the
// /interactions/{id} lifecycle) as typed Bifrost requests: governance, logging, pricing and
// fallbacks apply, which /genai_passthrough cannot offer.
//
// A request for Gemini or Vertex forwards the caller's body as it is and answers with Google's
// bytes (the native path), so agents, environments, webhooks and every built-in tool step
// survive. A request for any other provider (a provider/model prefix, a routing rule, or a
// fallback) is translated to the Responses schema and its answer rendered back as an Interaction.

// genAIInteractionsStreamStateKey holds the per-stream state of a translated stream.
const genAIInteractionsStreamStateKey schemas.BifrostContextKey = "bifrost-genai-interactions-stream-state"

// genAIInteractionsBasePaths are the Gemini API and Vertex shapes of the collection.
func genAIInteractionsBasePaths(pathPrefix string) []string {
	return []string{
		pathPrefix + "/v1beta/interactions",
		pathPrefix + "/v1/interactions",
		pathPrefix + "/v1/projects/{project}/locations/{location}/interactions",
		pathPrefix + "/v1beta1/projects/{project}/locations/{location}/interactions",
	}
}

// CreateGenAIInteractionsRouteConfigs creates the Interactions routes under pathPrefix.
func CreateGenAIInteractionsRouteConfigs(pathPrefix string) []RouteConfig {
	var routes []RouteConfig
	for _, base := range genAIInteractionsBasePaths(pathPrefix) {
		routes = append(routes,
			RouteConfig{
				Type:   RouteConfigTypeGenAI,
				Path:   base,
				Method: "POST",
				GetHTTPRequestType: func(ctx *fasthttp.RequestCtx) schemas.RequestType {
					return schemas.ResponsesRequest
				},
				GetRequestTypeInstance: func(ctx context.Context) interface{} {
					return &gemini.GeminiInteractionsRequest{}
				},
				PreCallback:                prepareGenAIInteractionsCreate,
				RequestConverter:           convertGenAIInteractionsCreate,
				ResponsesResponseConverter: genAIInteractionsResponseConverter,
				ErrorConverter:             genAIInteractionsErrorConverter,
				StreamConfig:               genAIInteractionsStreamConfig(),
			},
			RouteConfig{
				Type:   RouteConfigTypeGenAI,
				Path:   base + "/{interaction_id}",
				Method: "GET",
				GetHTTPRequestType: func(ctx *fasthttp.RequestCtx) schemas.RequestType {
					return schemas.ResponsesRetrieveRequest
				},
				GetRequestTypeInstance: func(ctx context.Context) interface{} {
					return &schemas.BifrostResponsesRetrieveRequest{}
				},
				PreCallback: prepareGenAIInteractionsLifecycle,
				RequestConverter: func(ctx *schemas.BifrostContext, req interface{}) (*schemas.BifrostRequest, error) {
					if r, ok := req.(*schemas.BifrostResponsesRetrieveRequest); ok {
						return &schemas.BifrostRequest{RequestType: schemas.ResponsesRetrieveRequest, ResponsesRetrieveRequest: r}, nil
					}
					return nil, errors.New("invalid interactions retrieve request")
				},
				ResponsesResponseConverter: genAIInteractionsResponseConverter,
				ErrorConverter:             genAIInteractionsErrorConverter,
				StreamConfig:               genAIInteractionsStreamConfig(),
			},
			RouteConfig{
				Type:   RouteConfigTypeGenAI,
				Path:   base + "/{interaction_id}",
				Method: "DELETE",
				GetHTTPRequestType: func(ctx *fasthttp.RequestCtx) schemas.RequestType {
					return schemas.ResponsesDeleteRequest
				},
				GetRequestTypeInstance: func(ctx context.Context) interface{} {
					return &schemas.BifrostResponsesDeleteRequest{}
				},
				PreCallback: prepareGenAIInteractionsLifecycle,
				RequestConverter: func(ctx *schemas.BifrostContext, req interface{}) (*schemas.BifrostRequest, error) {
					if r, ok := req.(*schemas.BifrostResponsesDeleteRequest); ok {
						return &schemas.BifrostRequest{RequestType: schemas.ResponsesDeleteRequest, ResponsesDeleteRequest: r}, nil
					}
					return nil, errors.New("invalid interactions delete request")
				},
				// Google answers a delete with an empty object.
				ResponsesDeleteResponseConverter: func(ctx *schemas.BifrostContext, resp *schemas.BifrostResponsesDeleteResponse) (interface{}, error) {
					return map[string]interface{}{}, nil
				},
				ErrorConverter: genAIInteractionsErrorConverter,
			},
			RouteConfig{
				Type:   RouteConfigTypeGenAI,
				Path:   base + "/{interaction_id}/cancel",
				Method: "POST",
				GetHTTPRequestType: func(ctx *fasthttp.RequestCtx) schemas.RequestType {
					return schemas.ResponsesCancelRequest
				},
				GetRequestTypeInstance: func(ctx context.Context) interface{} {
					return &schemas.BifrostResponsesCancelRequest{}
				},
				PreCallback: prepareGenAIInteractionsLifecycle,
				RequestConverter: func(ctx *schemas.BifrostContext, req interface{}) (*schemas.BifrostRequest, error) {
					if r, ok := req.(*schemas.BifrostResponsesCancelRequest); ok {
						return &schemas.BifrostRequest{RequestType: schemas.ResponsesCancelRequest, ResponsesCancelRequest: r}, nil
					}
					return nil, errors.New("invalid interactions cancel request")
				},
				ResponsesResponseConverter: genAIInteractionsResponseConverter,
				ErrorConverter:             genAIInteractionsErrorConverter,
			},
		)
	}
	return routes
}

// isGoogleInteractionsProvider reports whether provider serves the Interactions API natively.
func isGoogleInteractionsProvider(provider schemas.ModelProvider) bool {
	return provider == schemas.Gemini || provider == schemas.Vertex
}

// genAIInteractionsDefaultProvider is the provider a request that names none goes to: the
// x-model-provider header, else Vertex for a Vertex-shaped path or bearer token, else Gemini.
func genAIInteractionsDefaultProvider(ctx *fasthttp.RequestCtx, model string) schemas.ModelProvider {
	if header := getProviderFromHeader(ctx, ""); header != "" {
		return header
	}
	return detectProviderFromGenAIRequest(ctx, model)
}

// enableGenAIInteractionsNative turns on the native path: the caller's body goes upstream as it
// is (with its text fields still open to guardrail redaction) and Google's response comes back.
// Core turns the raw body off again for any attempt on a provider outside Google's family.
func enableGenAIInteractionsNative(bifrostCtx *schemas.BifrostContext) {
	bifrostCtx.SetValue(schemas.BifrostContextKeyUseRawRequestBody, true)
	bifrostCtx.SetValue(schemas.BifrostContextKeySendBackRawResponse, true)
	bifrostCtx.SetValue(
		schemas.BifrostContextKeyRawRequestBodyTextRewriter,
		schemas.RawRequestBodyTextRewriter(rewriteGenAIInteractionsRawRequestBody),
	)
}

// forwardGenAIInteractionsHeaders passes the Api-Revision header, which pins the Interactions
// API's response shape, through to the upstream.
func forwardGenAIInteractionsHeaders(ctx *fasthttp.RequestCtx, bifrostCtx *schemas.BifrostContext) {
	revision := strings.TrimSpace(string(ctx.Request.Header.Peek("Api-Revision")))
	if revision == "" {
		return
	}
	headers, _ := bifrostCtx.Value(schemas.BifrostContextKeyExtraHeaders).(map[string][]string)
	merged := make(map[string][]string, len(headers)+1)
	for k, v := range headers {
		merged[k] = v
	}
	merged["Api-Revision"] = []string{revision}
	bifrostCtx.SetValue(schemas.BifrostContextKeyExtraHeaders, merged)
}

func prepareGenAIInteractionsCreate(ctx *fasthttp.RequestCtx, bifrostCtx *schemas.BifrostContext, req interface{}) error {
	r, ok := req.(*gemini.GeminiInteractionsRequest)
	if !ok {
		return errors.New("invalid interactions request")
	}
	modelName := r.Model
	if modelName == "" {
		modelName = r.Agent
	}
	if modelName == "" {
		return errors.New("interactions request names neither a model nor an agent")
	}
	defaultProvider := genAIInteractionsDefaultProvider(ctx, modelName)
	provider, _ := schemas.ParseModelString(modelName, defaultProvider)
	bifrostCtx.SetValue(bifrostContextKeyProvider, defaultProvider)
	bifrostCtx.SetValue(schemas.BifrostContextKeyInteractionsAPI, true)

	headers := extractHeadersFromRequest(ctx)
	schemas.ExtractAndSetUserAgentFromHeaders(headers, bifrostCtx)
	forwardGenAIInteractionsHeaders(ctx, bifrostCtx)

	if isGoogleInteractionsProvider(provider) {
		enableGenAIInteractionsNative(bifrostCtx)
		return nil
	}
	return r.ValidateTranslatable()
}

func convertGenAIInteractionsCreate(ctx *schemas.BifrostContext, req interface{}) (*schemas.BifrostRequest, error) {
	r, ok := req.(*gemini.GeminiInteractionsRequest)
	if !ok {
		return nil, errors.New("invalid interactions request")
	}
	responsesReq, err := r.ToBifrostResponsesRequest()
	if err != nil {
		return nil, err
	}
	if responsesReq.Provider == "" {
		if provider, ok := ctx.Value(bifrostContextKeyProvider).(schemas.ModelProvider); ok && provider != "" {
			responsesReq.Provider = provider
		} else {
			responsesReq.Provider = schemas.Gemini
		}
	}
	return &schemas.BifrostRequest{RequestType: schemas.ResponsesRequest, ResponsesRequest: responsesReq}, nil
}

// prepareGenAIInteractionsLifecycle fills the interaction id and provider of a get, cancel or
// delete. The provider comes from ?provider=, the x-model-provider header, or the path and token
// shape, as on create; a get with ?stream=true replays the run, resuming after last_event_id.
func prepareGenAIInteractionsLifecycle(ctx *fasthttp.RequestCtx, bifrostCtx *schemas.BifrostContext, req interface{}) error {
	id, _ := ctx.UserValue("interaction_id").(string)
	if id == "" {
		return errors.New("interaction id is required")
	}
	provider := schemas.ModelProvider(string(ctx.QueryArgs().Peek("provider")))
	if provider == "" {
		provider = genAIInteractionsDefaultProvider(ctx, "")
	}
	bifrostCtx.SetValue(bifrostContextKeyProvider, provider)
	bifrostCtx.SetValue(schemas.BifrostContextKeyInteractionsAPI, true)
	forwardGenAIInteractionsHeaders(ctx, bifrostCtx)
	if isGoogleInteractionsProvider(provider) {
		bifrostCtx.SetValue(schemas.BifrostContextKeySendBackRawResponse, true)
		bifrostCtx.SetValue(schemas.BifrostContextKeyUseRawRequestBody, true)
	}

	switch r := req.(type) {
	case *schemas.BifrostResponsesRetrieveRequest:
		r.ResponseID, r.Provider = id, provider
		if raw := ctx.QueryArgs().Peek("stream"); len(raw) > 0 {
			stream := string(raw) == "true" || string(raw) == "1"
			r.Stream = &stream
		}
		if raw := ctx.QueryArgs().Peek("last_event_id"); len(raw) > 0 {
			last := string(raw)
			r.LastEventID = &last
		}
	case *schemas.BifrostResponsesDeleteRequest:
		r.ResponseID, r.Provider = id, provider
	case *schemas.BifrostResponsesCancelRequest:
		r.ResponseID, r.Provider = id, provider
	default:
		return errors.New("invalid interactions lifecycle request")
	}
	return nil
}

// genAIInteractionsNative reports whether a response came back on the native path: the attempt
// still forwarded the caller's body (core clears that for a provider outside Google's family) and
// the provider's own bytes are attached.
func genAIInteractionsNative(ctx *schemas.BifrostContext, raw interface{}) bool {
	if raw == nil {
		return false
	}
	useRaw, _ := ctx.Value(schemas.BifrostContextKeyUseRawRequestBody).(bool)
	return useRaw
}

func genAIInteractionsResponseConverter(ctx *schemas.BifrostContext, resp *schemas.BifrostResponsesResponse) (interface{}, error) {
	if genAIInteractionsNative(ctx, resp.ExtraFields.RawResponse) {
		return resp.ExtraFields.RawResponse, nil
	}
	return gemini.ToGeminiInteraction(resp), nil
}

func genAIInteractionsErrorConverter(ctx *schemas.BifrostContext, err *schemas.BifrostError) interface{} {
	return gemini.ToGeminiInteractionsErrorBody(err)
}

// genAIInteractionsDoneFrame closes an Interactions stream the way Google does.
const genAIInteractionsDoneFrame = "event: done\ndata: [DONE]\n\n"

func genAIInteractionsStreamConfig() *StreamConfig {
	return &StreamConfig{
		// Google's SDKs parse every line of an Interactions stream; a heartbeat comment would
		// break them, as on the generateContent route.
		ResponsesStreamResponseConverter: genAIInteractionsStreamConverter,
		ErrorConverter: func(ctx *schemas.BifrostContext, err *schemas.BifrostError) interface{} {
			body := gemini.ToGeminiInteractionsErrorBody(err)
			payload, marshalErr := providerUtils.MarshalSorted(map[string]interface{}{"error": body.Error, "event_type": gemini.GeminiInteractionsEventError})
			if marshalErr != nil {
				return body
			}
			return "event: error\ndata: " + string(payload) + "\n\n"
		},
	}
}

// genAIInteractionsStreamConverter writes one stream chunk. On the native path it replays the
// upstream event the chunk carries (chunks without one are skipped, so every event goes out
// exactly once); otherwise it renders the chunk as Interactions events.
func genAIInteractionsStreamConverter(ctx *schemas.BifrostContext, resp *schemas.BifrostResponsesStreamResponse) (string, interface{}, error) {
	if raw, ok := resp.ExtraFields.RawResponse.(string); ok && genAIInteractionsNative(ctx, resp.ExtraFields.RawResponse) {
		eventType := gjson.Get(raw, "event_type").String()
		frame := "event: " + eventType + "\ndata: " + raw + "\n\n"
		if eventType == gemini.GeminiInteractionsEventCompleted {
			frame += genAIInteractionsDoneFrame
		}
		return "", frame, nil
	}
	if useRaw, _ := ctx.Value(schemas.BifrostContextKeyUseRawRequestBody).(bool); useRaw && resp.ExtraFields.RawResponse == nil {
		// A native stream chunk that repeats an event already sent.
		return "", nil, nil
	}

	state, _ := ctx.Value(genAIInteractionsStreamStateKey).(*gemini.BifrostToGeminiInteractionsStreamState)
	if state == nil {
		state = gemini.NewBifrostToGeminiInteractionsStreamState()
		ctx.SetValue(genAIInteractionsStreamStateKey, state)
	}
	events := state.ToGeminiInteractionsStreamEvents(resp)
	if len(events) == 0 {
		return "", nil, nil
	}
	var sb strings.Builder
	for _, event := range events {
		payload, err := sonic.Marshal(event)
		if err != nil {
			return "", nil, fmt.Errorf("failed to encode interactions event: %w", err)
		}
		sb.WriteString("event: ")
		sb.WriteString(event.EventType)
		sb.WriteString("\ndata: ")
		sb.Write(payload)
		sb.WriteString("\n\n")
		if event.EventType == gemini.GeminiInteractionsEventCompleted {
			sb.WriteString(genAIInteractionsDoneFrame)
		}
	}
	return "", sb.String(), nil
}

// rewriteGenAIInteractionsRawRequestBody applies guardrail replacements to the text an
// Interactions body carries: the system instruction, a string input, and the text of content
// blocks and steps (including function results).
func rewriteGenAIInteractionsRawRequestBody(rawBody []byte, replacements map[string]string) ([]byte, error) {
	return rewriteRawRequestTextFields(rawBody, replacements, collectGenAIInteractionsTextPaths)
}

func collectGenAIInteractionsTextPaths(root gjson.Result, replacements map[string]string) ([]string, error) {
	var paths []string
	if err := appendRawRequestStringPath(&paths, root.Get("system_instruction"), "system_instruction"); err != nil {
		return nil, err
	}
	input := root.Get("input")
	switch {
	case !input.Exists() || input.Type == gjson.Null:
	case input.Type == gjson.String:
		paths = append(paths, "input")
	case input.IsObject():
		if err := collectGenAIInteractionsItemTextPaths(&paths, input, "input"); err != nil {
			return nil, err
		}
	case input.IsArray():
		var collectErr error
		input.ForEach(func(index, item gjson.Result) bool {
			collectErr = collectGenAIInteractionsItemTextPaths(&paths, item, rawRequestArrayPath("input", int(index.Int())))
			return collectErr == nil
		})
		if collectErr != nil {
			return nil, collectErr
		}
	default:
		return nil, fmt.Errorf("raw interactions input must be a string, an object, or an array")
	}
	return paths, nil
}

// collectGenAIInteractionsItemTextPaths adds the text paths of one content block or step.
func collectGenAIInteractionsItemTextPaths(paths *[]string, item gjson.Result, path string) error {
	if !item.IsObject() {
		return nil
	}
	if err := appendRawRequestStringPath(paths, item.Get("text"), rawRequestObjectPath(path, "text")); err != nil {
		return err
	}
	for _, key := range []string{"content", "summary", "result"} {
		child := item.Get(key)
		switch {
		case !child.Exists() || child.Type == gjson.Null:
		case child.Type == gjson.String && key == "result":
			*paths = append(*paths, rawRequestObjectPath(path, key))
		case child.IsArray():
			var collectErr error
			childPath := rawRequestObjectPath(path, key)
			child.ForEach(func(index, block gjson.Result) bool {
				collectErr = collectGenAIInteractionsItemTextPaths(paths, block, rawRequestArrayPath(childPath, int(index.Int())))
				return collectErr == nil
			})
			if collectErr != nil {
				return collectErr
			}
		}
	}
	if err := collectRawRequestArgumentStringPaths(paths, item.Get("arguments"), rawRequestObjectPath(path, "arguments")); err != nil {
		return err
	}
	return nil
}
