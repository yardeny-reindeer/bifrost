package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/bytedance/sonic"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func interactionsUserText(text string) schemas.ResponsesMessage {
	return userResponsesMessage(&schemas.ResponsesMessageContent{ContentStr: schemas.Ptr(text)})
}

func marshalInteractionsBody(t *testing.T, req *GeminiInteractionsRequest) string {
	t.Helper()
	raw, err := providerUtils.MarshalSorted(req)
	require.NoError(t, err)
	return string(raw)
}

// TestShouldUseGeminiInteractions pins the gate: only the Interactions route, a stored previous
// turn, a background run, or an agent leave generateContent.
func TestShouldUseGeminiInteractions(t *testing.T) {
	plain := &schemas.BifrostResponsesRequest{Model: "gemini-3.8-flash", Params: &schemas.ResponsesParameters{}}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	assert.False(t, ShouldUseGeminiInteractions(ctx, plain))
	assert.False(t, ShouldUseGeminiInteractions(ctx, &schemas.BifrostResponsesRequest{Model: "gemini-3.8-flash", Params: &schemas.ResponsesParameters{PreviousResponseID: schemas.Ptr("")}}))

	assert.True(t, ShouldUseGeminiInteractions(ctx, &schemas.BifrostResponsesRequest{Model: "gemini-3.8-flash", Params: &schemas.ResponsesParameters{PreviousResponseID: schemas.Ptr("v1_abc")}}))
	assert.True(t, ShouldUseGeminiInteractions(ctx, &schemas.BifrostResponsesRequest{Model: "gemini-3.8-flash", Params: &schemas.ResponsesParameters{Background: schemas.Ptr(true)}}))
	assert.True(t, ShouldUseGeminiInteractions(ctx, &schemas.BifrostResponsesRequest{Model: "deep-research-preview-04-2026"}))

	routed := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	routed.SetValue(schemas.BifrostContextKeyInteractionsAPI, true)
	assert.True(t, ShouldUseGeminiInteractions(routed, plain))
}

func TestToGeminiInteractionsRequestSingleTurn(t *testing.T) {
	req := &schemas.BifrostResponsesRequest{
		Model: "gemini-3.8-flash",
		Input: []schemas.ResponsesMessage{
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage), Role: schemas.Ptr(schemas.ResponsesInputMessageRoleDeveloper), Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("Be brief.")}},
			interactionsUserText("What is the capital of France?"),
		},
		Params: &schemas.ResponsesParameters{
			Instructions:       schemas.Ptr("You are a geographer."),
			PreviousResponseID: schemas.Ptr("v1_prev"),
			Store:              schemas.Ptr(true),
			Background:         schemas.Ptr(false),
			MaxOutputTokens:    schemas.Ptr(256),
			Temperature:        schemas.Ptr(0.2),
			Reasoning:          &schemas.ResponsesParametersReasoning{Effort: schemas.Ptr("high"), Summary: schemas.Ptr("auto")},
			ServiceTier:        schemas.Ptr(schemas.BifrostServiceTierFlex),
			ToolChoice:         &schemas.ResponsesToolChoice{ResponsesToolChoiceStruct: &schemas.ResponsesToolChoiceStruct{Type: schemas.ResponsesToolChoiceTypeFunction, Name: schemas.Ptr("get_weather")}},
			Tools: []schemas.ResponsesTool{
				{Type: schemas.ResponsesToolTypeFunction, Name: schemas.Ptr("get_weather"), Description: schemas.Ptr("Weather by city"), ResponsesToolFunction: &schemas.ResponsesToolFunction{Parameters: &schemas.ToolFunctionParameters{Type: "object", Required: []string{"city"}}}},
				{Type: schemas.ResponsesToolTypeWebSearch},
			},
			ExtraParams: map[string]interface{}{"seed": 7, "stop": []string{"END"}},
		},
	}
	native, err := ToGeminiInteractionsRequest(req)
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"model":"gemini-3.8-flash",
		"input":"What is the capital of France?",
		"system_instruction":"You are a geographer.\n\nBe brief.",
		"previous_interaction_id":"v1_prev",
		"store":true,
		"background":false,
		"service_tier":"flex",
		"tools":[
			{"type":"function","name":"get_weather","description":"Weather by city","parameters":{"type":"object","properties":{},"required":["city"]}},
			{"type":"google_search"}
		],
		"generation_config":{
			"max_output_tokens":256,"temperature":0.2,"seed":7,"stop_sequences":["END"],
			"thinking_level":"high","thinking_summaries":"auto",
			"tool_choice":{"allowed_tools":{"mode":"any","tools":["get_weather"]}}
		}
	}`, marshalInteractionsBody(t, native))
	assert.Nil(t, native.ExtraParams, "seed and stop are moved into generation_config")
}

// TestToGeminiInteractionsRequestHistoryAsSteps pins stateless replay: a history with model turns,
// reasoning and tool traffic is sent as steps, with the thought signature kept.
func TestToGeminiInteractionsRequestHistoryAsSteps(t *testing.T) {
	req := &schemas.BifrostResponsesRequest{
		Model: "gemini-3.8-flash",
		Input: []schemas.ResponsesMessage{
			interactionsUserText("Weather in Paris?"),
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeReasoning), ResponsesReasoning: &schemas.ResponsesReasoning{EncryptedContent: schemas.Ptr("sig_1"), Summary: []schemas.ResponsesReasoningSummary{{Type: schemas.ResponsesReasoningContentBlockTypeSummaryText, Text: "Need weather."}}}},
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall), ResponsesToolMessage: &schemas.ResponsesToolMessage{CallID: schemas.Ptr("call_1"), Name: schemas.Ptr("get_weather"), Arguments: schemas.Ptr(`{"city":"Paris"}`)}},
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCallOutput), ResponsesToolMessage: &schemas.ResponsesToolMessage{CallID: schemas.Ptr("call_1"), Name: schemas.Ptr("get_weather"), Output: &schemas.ResponsesToolMessageOutputStruct{ResponsesToolCallOutputStr: schemas.Ptr(`{"temp":21}`)}}},
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage), Role: schemas.Ptr(schemas.ResponsesInputMessageRoleAssistant), Content: &schemas.ResponsesMessageContent{ContentBlocks: []schemas.ResponsesMessageContentBlock{{Type: schemas.ResponsesOutputMessageContentTypeText, Text: schemas.Ptr("It is 21C.")}}}},
			interactionsUserText("Thanks"),
		},
	}
	native, err := ToGeminiInteractionsRequest(req)
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"model":"gemini-3.8-flash",
		"input":[
			{"type":"user_input","content":[{"type":"text","text":"Weather in Paris?"}]},
			{"type":"thought","signature":"sig_1","summary":[{"type":"text","text":"Need weather."}]},
			{"type":"function_call","id":"call_1","name":"get_weather","arguments":{"city":"Paris"}},
			{"type":"function_result","call_id":"call_1","name":"get_weather","result":[{"type":"text","text":"{\"temp\":21}"}]},
			{"type":"model_output","content":[{"type":"text","text":"It is 21C."}]},
			{"type":"user_input","content":[{"type":"text","text":"Thanks"}]}
		]
	}`, marshalInteractionsBody(t, native))
}

func TestToGeminiInteractionsRequestAgentAndMedia(t *testing.T) {
	req := &schemas.BifrostResponsesRequest{
		Model: "deep-research-preview-04-2026",
		Input: []schemas.ResponsesMessage{userResponsesMessage(&schemas.ResponsesMessageContent{ContentBlocks: []schemas.ResponsesMessageContentBlock{
			{Type: schemas.ResponsesInputMessageContentBlockTypeText, Text: schemas.Ptr("Summarize")},
			{Type: schemas.ResponsesInputMessageContentBlockTypeImage, ResponsesInputMessageContentBlockImage: &schemas.ResponsesInputMessageContentBlockImage{ImageURL: schemas.Ptr("data:image/png;base64,AAAA")}},
			{Type: schemas.ResponsesInputMessageContentBlockTypeFile, ResponsesInputMessageContentBlockFile: &schemas.ResponsesInputMessageContentBlockFile{FileURL: schemas.Ptr("https://example.com/report.pdf")}},
		}})},
		Params: &schemas.ResponsesParameters{Background: schemas.Ptr(true), Text: &schemas.ResponsesTextConfig{Format: &schemas.ResponsesTextConfigFormat{Type: "json_object"}}},
	}
	native, err := ToGeminiInteractionsRequest(req)
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"agent":"deep-research-preview-04-2026",
		"background":true,
		"input":[
			{"type":"text","text":"Summarize"},
			{"type":"image","data":"AAAA","mime_type":"image/png"},
			{"type":"document","uri":"https://example.com/report.pdf","mime_type":"application/pdf"}
		],
		"response_format":{"type":"text","mime_type":"application/json"}
	}`, marshalInteractionsBody(t, native))
}

func TestToGeminiInteractionsRequestRefusesUnservedTool(t *testing.T) {
	req := &schemas.BifrostResponsesRequest{
		Model:  "gemini-3.8-flash",
		Input:  []schemas.ResponsesMessage{interactionsUserText("hi")},
		Params: &schemas.ResponsesParameters{Tools: []schemas.ResponsesTool{{Type: schemas.ResponsesToolTypeLocalShell}}},
	}
	_, err := ToGeminiInteractionsRequest(req)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `"local_shell"`)
}

func TestGeminiInteractionsInputUnmarshal(t *testing.T) {
	var req GeminiInteractionsRequest
	require.NoError(t, sonic.Unmarshal([]byte(`{"model":"m","input":"hello"}`), &req))
	require.NotNil(t, req.Input.Text)
	assert.Equal(t, "hello", *req.Input.Text)

	req = GeminiInteractionsRequest{}
	require.NoError(t, sonic.Unmarshal([]byte(`{"model":"m","input":{"type":"function_result","call_id":"c1","result":"ok"}}`), &req))
	require.Len(t, req.Input.Items, 1)
	assert.Equal(t, "function_result", req.Input.Items[0].Type)

	req = GeminiInteractionsRequest{}
	require.NoError(t, sonic.Unmarshal([]byte(`{"model":"m","input":[{"type":"text","text":"a"},{"type":"text","text":"b"}],"stream":true}`), &req))
	assert.Len(t, req.Input.Items, 2)
	assert.True(t, req.IsStreamingRequested())
}

// TestGeminiInteractionsRequestToBifrost pins the inbound translation used when an Interactions
// caller's request runs on another provider.
func TestGeminiInteractionsRequestToBifrost(t *testing.T) {
	body := `{
		"model":"openai/gpt-5",
		"system_instruction":"Be brief.",
		"previous_interaction_id":"resp_1",
		"input":[
			{"type":"function_result","call_id":"call_1","name":"get_weather","result":[{"type":"text","text":"21C"}]},
			{"type":"text","text":"and tomorrow?"}
		],
		"tools":[{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}],
		"generation_config":{"max_output_tokens":100,"thinking_level":"low","tool_choice":"any","seed":3},
		"response_format":{"type":"text","mime_type":"application/json","schema":{"type":"object"}},
		"service_tier":"priority",
		"fallbacks":["anthropic/claude-sonnet-5-5"]
	}`
	var req GeminiInteractionsRequest
	require.NoError(t, sonic.Unmarshal([]byte(body), &req))
	require.NoError(t, req.ValidateTranslatable())
	out, err := req.ToBifrostResponsesRequest()
	require.NoError(t, err)
	assert.Equal(t, schemas.OpenAI, out.Provider)
	assert.Equal(t, "gpt-5", out.Model)
	require.Len(t, out.Fallbacks, 1)
	assert.Equal(t, schemas.Anthropic, out.Fallbacks[0].Provider)

	p := out.Params
	assert.Equal(t, "Be brief.", *p.Instructions)
	assert.Equal(t, "resp_1", *p.PreviousResponseID)
	assert.Equal(t, 100, *p.MaxOutputTokens)
	assert.Equal(t, "low", *p.Reasoning.Effort)
	assert.Equal(t, "required", *p.ToolChoice.ResponsesToolChoiceStr)
	assert.Equal(t, 3, p.ExtraParams["seed"])
	assert.Equal(t, schemas.BifrostServiceTierPriority, *p.ServiceTier)
	assert.Equal(t, "json_schema", p.Text.Format.Type)
	require.Len(t, p.Tools, 1)
	assert.Equal(t, "get_weather", *p.Tools[0].Name)

	require.Len(t, out.Input, 2)
	assert.Equal(t, schemas.ResponsesMessageTypeFunctionCallOutput, *out.Input[0].Type)
	assert.Equal(t, "call_1", *out.Input[0].CallID)
	assert.Equal(t, "21C", *out.Input[0].Output.ResponsesToolCallOutputStr)
	assert.Equal(t, schemas.ResponsesInputMessageRoleUser, *out.Input[1].Role)
	assert.Equal(t, "and tomorrow?", *out.Input[1].Content.ContentBlocks[0].Text)
}

func TestGeminiInteractionsRequestNativeOnlyFields(t *testing.T) {
	var req GeminiInteractionsRequest
	require.NoError(t, sonic.Unmarshal([]byte(`{"agent":"deep-research-preview-04-2026","input":"go","agent_config":{"type":"deep-research"},"background":true}`), &req))
	err := req.ValidateTranslatable()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "agent")
	assert.Contains(t, err.Error(), "agent_config")

	out, convErr := req.ToBifrostResponsesRequest()
	require.NoError(t, convErr)
	assert.Equal(t, "deep-research-preview-04-2026", out.Model)
	assert.JSONEq(t, `{"type":"deep-research"}`, string(out.Params.ExtraParams["agent_config"].(json.RawMessage)))
}

func TestGeminiInteractionToBifrostResponses(t *testing.T) {
	body := `{
		"id":"v1_run","object":"interaction","model":"gemini-3.8-flash","status":"requires_action",
		"created":"2026-11-18T10:00:00Z","updated":"2026-11-18T10:00:05Z",
		"steps":[
			{"type":"thought","signature":"sig_a","summary":[{"type":"text","text":"Thinking"}]},
			{"type":"google_search_call","id":"s1","arguments":{"queries":["paris weather"]}},
			{"type":"model_output","content":[{"type":"text","text":"Checking."}]},
			{"type":"function_call","id":"call_9","name":"get_weather","arguments":{"city":"Paris"}}
		],
		"usage":{"total_input_tokens":10,"total_output_tokens":5,"total_thought_tokens":7,"total_cached_tokens":2,"total_tokens":22}
	}`
	var interaction GeminiInteraction
	require.NoError(t, sonic.Unmarshal([]byte(body), &interaction))
	resp := interaction.ToBifrostResponsesResponse()

	assert.Equal(t, "v1_run", *resp.ID)
	assert.Equal(t, "gemini-3.8-flash", resp.Model)
	assert.Equal(t, schemas.ResponsesResponseStatusCompleted, *resp.Status, "requires_action is a finished turn waiting on function results")
	require.Len(t, resp.Output, 4)
	assert.Equal(t, schemas.ResponsesMessageTypeReasoning, *resp.Output[0].Type)
	assert.Equal(t, "sig_a", *resp.Output[0].ResponsesReasoning.EncryptedContent)
	assert.Equal(t, schemas.ResponsesMessageTypeWebSearchCall, *resp.Output[1].Type, "a search with no result yet is still its item")
	assert.Equal(t, "in_progress", *resp.Output[1].Status)
	assert.Equal(t, schemas.ResponsesMessageTypeMessage, *resp.Output[2].Type)
	assert.Equal(t, "Checking.", *resp.Output[2].Content.ContentBlocks[0].Text)
	assert.Equal(t, schemas.ResponsesMessageTypeFunctionCall, *resp.Output[3].Type)
	assert.Equal(t, "call_9", *resp.Output[3].CallID)
	assert.JSONEq(t, `{"city":"Paris"}`, *resp.Output[3].Arguments)
	assert.Nil(t, resp.ProviderExtraFields["steps"], "every step here has a Responses item")

	require.NotNil(t, resp.Usage)
	assert.Equal(t, 10, resp.Usage.InputTokens)
	assert.Equal(t, 12, resp.Usage.OutputTokens, "thought tokens are output")
	assert.Equal(t, 7, resp.Usage.OutputTokensDetails.ReasoningTokens)
	assert.Equal(t, 2, resp.Usage.InputTokensDetails.CachedReadTokens)
	assert.Equal(t, 22, resp.Usage.TotalTokens)
}

func TestToGeminiInteractionFromResponses(t *testing.T) {
	resp := &schemas.BifrostResponsesResponse{
		ID: schemas.Ptr("resp_1"), Model: "gpt-5", CreatedAt: 1700000000, Status: schemas.Ptr("completed"),
		Output: []schemas.ResponsesMessage{
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage), Role: schemas.Ptr(schemas.ResponsesInputMessageRoleAssistant), Content: &schemas.ResponsesMessageContent{ContentBlocks: []schemas.ResponsesMessageContentBlock{{Type: schemas.ResponsesOutputMessageContentTypeText, Text: schemas.Ptr("Calling")}}}},
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall), ResponsesToolMessage: &schemas.ResponsesToolMessage{CallID: schemas.Ptr("call_1"), Name: schemas.Ptr("f"), Arguments: schemas.Ptr(`{"a":1}`)}},
		},
		Usage: &schemas.ResponsesResponseUsage{InputTokens: 3, OutputTokens: 9, TotalTokens: 12, OutputTokensDetails: &schemas.ResponsesResponseOutputTokens{ReasoningTokens: 4}},
	}
	out := ToGeminiInteraction(resp)
	raw, err := providerUtils.MarshalSorted(out)
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"id":"resp_1","object":"interaction","model":"gpt-5","status":"requires_action",
		"created":"2023-11-14T22:13:20Z","updated":"2023-11-14T22:13:20Z",
		"steps":[
			{"type":"model_output","content":[{"type":"text","text":"Calling"}]},
			{"type":"function_call","id":"call_1","name":"f","arguments":{"a":1}}
		],
		"usage":{"total_input_tokens":3,"total_output_tokens":5,"total_thought_tokens":4,"total_tokens":12}
	}`, string(raw))
}

func interactionsEvents(t *testing.T, lines ...string) []*GeminiInteractionsEvent {
	t.Helper()
	events := make([]*GeminiInteractionsEvent, 0, len(lines))
	for _, line := range lines {
		event, err := ParseGeminiInteractionsEvent([]byte(line))
		require.NoError(t, err)
		events = append(events, event)
	}
	return events
}

// TestGeminiInteractionsStreamToBifrost pins the SSE mapping: steps open and close output items,
// text, thought summaries and argument fragments become their delta events, and the completed
// event carries the run's usage.
func TestGeminiInteractionsStreamToBifrost(t *testing.T) {
	events := interactionsEvents(t,
		`{"event_type":"interaction.created","interaction":{"id":"v1_s","status":"in_progress","model":"gemini-3.8-flash"}}`,
		`{"event_type":"interaction.status_update","interaction_id":"v1_s","status":"in_progress"}`,
		`{"event_type":"step.start","index":0,"step":{"type":"thought"}}`,
		`{"event_type":"step.delta","index":0,"delta":{"type":"thought_summary","content":{"type":"text","text":"Plan"}}}`,
		`{"event_type":"step.delta","index":0,"delta":{"type":"thought_signature","signature":"sig_z"}}`,
		`{"event_type":"step.stop","index":0}`,
		`{"event_type":"step.start","index":1,"step":{"type":"model_output"}}`,
		`{"event_type":"step.delta","index":1,"delta":{"type":"text","text":"Hel"}}`,
		`{"event_type":"step.delta","index":1,"delta":{"type":"text","text":"lo"}}`,
		`{"event_type":"step.stop","index":1}`,
		`{"event_type":"step.start","index":2,"step":{"type":"function_call","id":"call_5","name":"f","arguments":{}}}`,
		`{"event_type":"step.delta","index":2,"delta":{"type":"arguments_delta","arguments":"{\"a\":"}}`,
		`{"event_type":"step.delta","index":2,"delta":{"type":"arguments_delta","arguments":"1}"}}`,
		`{"event_type":"step.stop","index":2}`,
		`{"event_type":"interaction.completed","interaction":{"id":"v1_s","status":"requires_action","model":"gemini-3.8-flash","usage":{"total_input_tokens":4,"total_output_tokens":6,"total_thought_tokens":2,"total_tokens":12}}}`,
	)
	state := NewGeminiInteractionsStreamState("gemini-3.8-flash")
	var types []schemas.ResponsesStreamResponseType
	var last *schemas.BifrostResponsesStreamResponse
	perEvent := make([]int, len(events))
	for i, event := range events {
		out, err := state.ToBifrostResponsesStream(event)
		require.Nil(t, err)
		perEvent[i] = len(out)
		for _, ev := range out {
			types = append(types, ev.Type)
			last = ev
			if ev.Type == schemas.ResponsesStreamResponseTypeOutputItemDone && ev.Item != nil && ev.Item.Type != nil {
				switch *ev.Item.Type {
				case schemas.ResponsesMessageTypeReasoning:
					assert.Equal(t, "sig_z", *ev.Item.ResponsesReasoning.EncryptedContent)
					assert.Equal(t, "Plan", ev.Item.ResponsesReasoning.Summary[0].Text)
				case schemas.ResponsesMessageTypeMessage:
					assert.Equal(t, "Hello", *ev.Item.Content.ContentBlocks[0].Text)
				case schemas.ResponsesMessageTypeFunctionCall:
					assert.Equal(t, `{"a":1}`, *ev.Item.Arguments)
					assert.Equal(t, "call_5", *ev.Item.CallID)
				}
			}
		}
	}
	assert.Equal(t, 0, perEvent[1], "a status update between steps has no Responses event")
	assert.Equal(t, []schemas.ResponsesStreamResponseType{
		schemas.ResponsesStreamResponseTypeCreated, schemas.ResponsesStreamResponseTypeInProgress,
		schemas.ResponsesStreamResponseTypeOutputItemAdded,
		schemas.ResponsesStreamResponseTypeReasoningSummaryPartAdded, schemas.ResponsesStreamResponseTypeReasoningSummaryTextDelta,
		schemas.ResponsesStreamResponseTypeReasoningSummaryTextDone, schemas.ResponsesStreamResponseTypeReasoningSummaryPartDone, schemas.ResponsesStreamResponseTypeOutputItemDone,
		schemas.ResponsesStreamResponseTypeOutputItemAdded,
		schemas.ResponsesStreamResponseTypeContentPartAdded, schemas.ResponsesStreamResponseTypeOutputTextDelta, schemas.ResponsesStreamResponseTypeOutputTextDelta,
		schemas.ResponsesStreamResponseTypeOutputTextDone, schemas.ResponsesStreamResponseTypeContentPartDone, schemas.ResponsesStreamResponseTypeOutputItemDone,
		schemas.ResponsesStreamResponseTypeOutputItemAdded,
		schemas.ResponsesStreamResponseTypeFunctionCallArgumentsDelta, schemas.ResponsesStreamResponseTypeFunctionCallArgumentsDelta,
		schemas.ResponsesStreamResponseTypeFunctionCallArgumentsDone, schemas.ResponsesStreamResponseTypeOutputItemDone,
		schemas.ResponsesStreamResponseTypeCompleted,
	}, types)
	require.NotNil(t, last.Response)
	assert.Equal(t, "v1_s", *last.Response.ID)
	assert.Equal(t, 8, last.Response.Usage.OutputTokens)
	assert.Equal(t, 12, last.Response.Usage.TotalTokens)
}

func TestGeminiInteractionsStreamError(t *testing.T) {
	state := NewGeminiInteractionsStreamState("m")
	_, err := state.ToBifrostResponsesStream(interactionsEvents(t, `{"event_type":"error","error":{"code":"resource_exhausted","message":"quota"}}`)[0])
	require.NotNil(t, err)
	assert.Equal(t, "quota", err.Error.Message)
	assert.Equal(t, "resource_exhausted", *err.Error.Code)
}

// TestBifrostToGeminiInteractionsStream pins the reverse rendering for a translated stream.
func TestBifrostToGeminiInteractionsStream(t *testing.T) {
	idx := 0
	state := NewBifrostToGeminiInteractionsStreamState()
	var got []string
	for _, ev := range []*schemas.BifrostResponsesStreamResponse{
		{Type: schemas.ResponsesStreamResponseTypeCreated, Response: &schemas.BifrostResponsesResponse{ID: schemas.Ptr("resp_9"), Model: "gpt-5"}},
		{Type: schemas.ResponsesStreamResponseTypeInProgress},
		{Type: schemas.ResponsesStreamResponseTypeOutputItemAdded, OutputIndex: &idx, Item: &schemas.ResponsesMessage{Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage), Role: schemas.Ptr(schemas.ResponsesInputMessageRoleAssistant)}},
		{Type: schemas.ResponsesStreamResponseTypeOutputTextDelta, OutputIndex: &idx, Delta: schemas.Ptr("Hi")},
		{Type: schemas.ResponsesStreamResponseTypeOutputItemDone, OutputIndex: &idx},
		{Type: schemas.ResponsesStreamResponseTypeCompleted, Response: &schemas.BifrostResponsesResponse{ID: schemas.Ptr("resp_9"), Model: "gpt-5", Status: schemas.Ptr("completed"), Usage: &schemas.ResponsesResponseUsage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2}}},
	} {
		for _, out := range state.ToGeminiInteractionsStreamEvents(ev) {
			raw, err := sonic.Marshal(out)
			require.NoError(t, err)
			got = append(got, string(raw))
		}
	}
	require.Len(t, got, 5)
	assert.JSONEq(t, `{"event_type":"interaction.created","event_id":"evt_1","interaction":{"id":"resp_9","object":"interaction","model":"gpt-5","status":"in_progress"}}`, got[0])
	assert.JSONEq(t, `{"event_type":"step.start","event_id":"evt_2","index":0,"step":{"type":"model_output"}}`, got[1])
	assert.JSONEq(t, `{"event_type":"step.delta","event_id":"evt_3","index":0,"delta":{"type":"text","text":"Hi"}}`, got[2])
	assert.JSONEq(t, `{"event_type":"step.stop","event_id":"evt_4","index":0}`, got[3])
	assert.Contains(t, got[4], `"event_type":"interaction.completed"`)
	assert.Contains(t, got[4], `"total_tokens":2`)
}

func TestNormalizeRawGeminiInteractionsBody(t *testing.T) {
	out, err := normalizeRawGeminiInteractionsBody([]byte(`{"model":"gemini/gemini-3.8-flash","input":"hi","fallbacks":["vertex/gemini-3.8-flash"]}`), "gemini-3.8-flash")
	require.Nil(t, err)
	assert.JSONEq(t, `{"model":"gemini-3.8-flash","input":"hi"}`, string(out))

	out, err = normalizeRawGeminiInteractionsBody([]byte(`{"agent":"gemini/deep-research-preview-04-2026","input":"go"}`), "deep-research-preview-04-2026")
	require.Nil(t, err)
	assert.JSONEq(t, `{"agent":"deep-research-preview-04-2026","input":"go"}`, string(out))
}

// newInteractionsTestServer answers Interactions calls with body and records each request.
type interactionsCall struct {
	method, path, query, apiKey, body string
}

func newInteractionsTestServer(t *testing.T, contentType string, reply func(call interactionsCall) string) (*httptest.Server, *[]interactionsCall) {
	t.Helper()
	calls := &[]interactionsCall{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		call := interactionsCall{method: r.Method, path: r.URL.Path, query: r.URL.RawQuery, apiKey: r.Header.Get("x-goog-api-key"), body: string(raw)}
		*calls = append(*calls, call)
		w.Header().Set("Content-Type", contentType)
		_, _ = io.WriteString(w, reply(call))
	}))
	t.Cleanup(server.Close)
	return server, calls
}

// TestGeminiResponsesPreviousResponseIDUsesInteractions pins the egress switch: a Responses request
// naming a previous response goes to POST {base}/interactions with previous_interaction_id, and the
// interaction id comes back as the response id.
func TestGeminiResponsesPreviousResponseIDUsesInteractions(t *testing.T) {
	server, calls := newInteractionsTestServer(t, "application/json", func(interactionsCall) string {
		return `{"id":"v1_next","object":"interaction","model":"gemini-3.8-flash","status":"completed","steps":[{"type":"model_output","content":[{"type":"text","text":"Paris."}]}],"usage":{"total_input_tokens":5,"total_output_tokens":2,"total_tokens":7}}`
	})
	provider := NewGeminiProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL + "/v1beta"}}, nil)
	key := schemas.Key{Value: schemas.SecretVar{Val: "test-key"}, Models: []string{"*"}}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	resp, bifrostErr := provider.Responses(ctx, key, &schemas.BifrostResponsesRequest{
		Provider: schemas.Gemini,
		Model:    "gemini-3.8-flash",
		Input:    []schemas.ResponsesMessage{interactionsUserText("And its capital?")},
		Params:   &schemas.ResponsesParameters{PreviousResponseID: schemas.Ptr("v1_prev")},
	})
	require.Nil(t, bifrostErr)
	require.Len(t, *calls, 1)
	assert.Equal(t, "/v1beta/interactions", (*calls)[0].path)
	assert.Equal(t, "test-key", (*calls)[0].apiKey)
	assert.JSONEq(t, `{"model":"gemini-3.8-flash","input":"And its capital?","previous_interaction_id":"v1_prev"}`, (*calls)[0].body)
	assert.Equal(t, "v1_next", *resp.ID)
	assert.Equal(t, "Paris.", *resp.Output[0].Content.ContentBlocks[0].Text)
	assert.Equal(t, 7, resp.Usage.TotalTokens)
	assert.Nil(t, resp.ProviderJob, "a run that finished in the foreground is billed on its own row")
}

// TestGeminiResponsesBackgroundRecordsJob pins billing for a background run: the create reports
// the run on ProviderJob so job accounting settles it once it ends.
func TestGeminiResponsesBackgroundRecordsJob(t *testing.T) {
	server, _ := newInteractionsTestServer(t, "application/json", func(interactionsCall) string {
		return `{"id":"v1_bg","object":"interaction","agent":"deep-research-preview-04-2026","status":"in_progress"}`
	})
	provider := NewGeminiProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL + "/v1beta"}}, nil)
	key := schemas.Key{Value: schemas.SecretVar{Val: "k"}, Models: []string{"*"}}
	resp, bifrostErr := provider.Responses(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), key, &schemas.BifrostResponsesRequest{
		Model:  "deep-research-preview-04-2026",
		Input:  []schemas.ResponsesMessage{interactionsUserText("Research X")},
		Params: &schemas.ResponsesParameters{Background: schemas.Ptr(true)},
	})
	require.Nil(t, bifrostErr)
	assert.Equal(t, schemas.ResponsesResponseStatusInProgress, *resp.Status)
	require.NotNil(t, resp.ProviderJob)
	assert.Equal(t, "v1_bg", resp.ProviderJob.ID)
	assert.True(t, resp.ProviderJob.Created)
	assert.False(t, resp.ProviderJob.Terminal)
	assert.Equal(t, "/interactions/v1_bg", resp.ProviderJob.PollPath)
}

// TestGeminiResponsesPlainRequestKeepsGenerateContent pins that the common case is untouched.
func TestGeminiResponsesPlainRequestKeepsGenerateContent(t *testing.T) {
	server, calls := newInteractionsTestServer(t, "application/json", func(interactionsCall) string {
		return `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`
	})
	provider := NewGeminiProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL + "/v1beta"}}, nil)
	key := schemas.Key{Value: schemas.SecretVar{Val: "k"}, Models: []string{"*"}}
	_, bifrostErr := provider.Responses(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), key, &schemas.BifrostResponsesRequest{
		Model: "gemini-3.8-flash",
		Input: []schemas.ResponsesMessage{interactionsUserText("hi")},
	})
	require.Nil(t, bifrostErr)
	require.Len(t, *calls, 1)
	assert.Equal(t, "/v1beta/models/gemini-3.8-flash:generateContent", (*calls)[0].path)
}

// TestGeminiResponsesLifecycle pins retrieve, cancel and delete on /interactions/{id}.
func TestGeminiResponsesLifecycle(t *testing.T) {
	server, calls := newInteractionsTestServer(t, "application/json", func(call interactionsCall) string {
		switch {
		case call.method == http.MethodDelete:
			return `{}`
		case strings.HasSuffix(call.path, "/cancel"):
			return `{"id":"v1_x","status":"cancelled"}`
		}
		return `{"id":"v1_x","status":"completed","model":"gemini-3.8-flash","steps":[{"type":"user_input","content":[{"type":"text","text":"q"}]},{"type":"model_output","content":[{"type":"text","text":"a"}]}],"usage":{"total_input_tokens":1,"total_output_tokens":1,"total_tokens":2}}`
	})
	provider := NewGeminiProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL + "/v1beta"}}, nil)
	key := schemas.Key{Value: schemas.SecretVar{Val: "k"}, Models: []string{"*"}}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	got, bifrostErr := provider.ResponsesRetrieve(ctx, key, &schemas.BifrostResponsesRetrieveRequest{ResponseID: "v1_x"})
	require.Nil(t, bifrostErr)
	require.Len(t, got.Output, 1, "user_input steps are input, not output")
	require.NotNil(t, got.ProviderJob)
	assert.False(t, got.ProviderJob.Created)
	assert.True(t, got.ProviderJob.Terminal)
	assert.Equal(t, 2, got.ProviderJob.Usage.LLMUsage.TotalTokens)

	cancelled, bifrostErr := provider.ResponsesCancel(ctx, key, &schemas.BifrostResponsesCancelRequest{ResponseID: "v1_x"})
	require.Nil(t, bifrostErr)
	assert.Equal(t, schemas.ResponsesResponseStatusCancelled, *cancelled.Status)

	deleted, bifrostErr := provider.ResponsesDelete(ctx, key, &schemas.BifrostResponsesDeleteRequest{ResponseID: "v1_x"})
	require.Nil(t, bifrostErr)
	assert.True(t, deleted.Deleted)

	items, bifrostErr := provider.ResponsesInputItems(ctx, key, &schemas.BifrostResponsesInputItemsRequest{ResponseID: "v1_x"})
	require.Nil(t, bifrostErr)
	require.Len(t, items.Data, 1)
	assert.Equal(t, "q", *items.Data[0].Content.ContentBlocks[0].Text)

	require.Len(t, *calls, 4)
	assert.Equal(t, "GET /v1beta/interactions/v1_x", (*calls)[0].method+" "+(*calls)[0].path)
	assert.Equal(t, "POST /v1beta/interactions/v1_x/cancel", (*calls)[1].method+" "+(*calls)[1].path)
	assert.Equal(t, "DELETE /v1beta/interactions/v1_x", (*calls)[2].method+" "+(*calls)[2].path)
}

// TestGeminiResponsesStreamViaInteractions pins the streamed create: stream:true in the body,
// alt=sse on the URL, Google's SSE framing parsed, and each raw event attached to one chunk for
// the native route (an event with no Responses counterpart rides a raw-only chunk).
func TestGeminiResponsesStreamViaInteractions(t *testing.T) {
	sse := strings.Join([]string{
		"event: interaction.created", `data: {"event_type":"interaction.created","interaction":{"id":"v1_s","status":"in_progress","model":"gemini-3.8-flash"}}`, "",
		"event: interaction.status_update", `data: {"event_type":"interaction.status_update","interaction_id":"v1_s","status":"in_progress"}`, "",
		"event: step.start", `data: {"event_type":"step.start","index":0,"step":{"type":"model_output"}}`, "",
		"event: step.delta", `data: {"event_type":"step.delta","index":0,"delta":{"type":"text","text":"Hi"}}`, "",
		"event: step.stop", `data: {"event_type":"step.stop","index":0}`, "",
		"event: interaction.completed", `data: {"event_type":"interaction.completed","interaction":{"id":"v1_s","status":"completed","usage":{"total_input_tokens":1,"total_output_tokens":1,"total_tokens":2}}}`, "",
		"event: done", "data: [DONE]", "",
	}, "\n")
	server, calls := newInteractionsTestServer(t, "text/event-stream", func(interactionsCall) string { return sse })
	provider := NewGeminiProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL + "/v1beta"}}, nil)
	key := schemas.Key{Value: schemas.SecretVar{Val: "k"}, Models: []string{"*"}}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeyInteractionsAPI, true)
	ctx.SetValue(schemas.BifrostContextKeyCaptureRawResponse, true) // what core derives from the route's send-back-raw flag

	postHook := func(ctx *schemas.BifrostContext, result *schemas.BifrostResponse, err *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError) {
		return result, err
	}
	stream, bifrostErr := provider.ResponsesStream(ctx, postHook, nil, key, &schemas.BifrostResponsesRequest{Model: "gemini-3.8-flash", Input: []schemas.ResponsesMessage{interactionsUserText("hi")}})
	require.Nil(t, bifrostErr)

	var rawEvents []string
	var types []schemas.ResponsesStreamResponseType
	for chunk := range stream {
		require.Nil(t, chunk.BifrostError)
		resp := chunk.BifrostResponsesStreamResponse
		require.NotNil(t, resp)
		types = append(types, resp.Type)
		if raw, ok := resp.ExtraFields.RawResponse.(string); ok {
			rawEvents = append(rawEvents, raw)
		}
	}
	require.Len(t, *calls, 1)
	assert.Equal(t, "/v1beta/interactions", (*calls)[0].path)
	assert.Equal(t, "alt=sse", (*calls)[0].query)
	assert.JSONEq(t, `{"model":"gemini-3.8-flash","input":"hi","stream":true}`, (*calls)[0].body)
	assert.Len(t, rawEvents, 6, "every upstream event reaches a native caller exactly once")
	assert.Contains(t, types, schemas.ResponsesStreamResponseTypeProviderRawEvent)
	assert.Equal(t, schemas.ResponsesStreamResponseTypeCompleted, types[len(types)-1])
}

// BenchmarkToGeminiInteractionsRequest measures converting a multi-turn Responses request with a
// tool into an Interactions body, the work the /v1/responses -> Gemini Interactions path adds.
func BenchmarkToGeminiInteractionsRequest(b *testing.B) {
	req := &schemas.BifrostResponsesRequest{
		Model: "gemini-3.8-flash",
		Input: []schemas.ResponsesMessage{
			interactionsUserText("Weather in Paris?"),
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall), ResponsesToolMessage: &schemas.ResponsesToolMessage{CallID: schemas.Ptr("call_1"), Name: schemas.Ptr("get_weather"), Arguments: schemas.Ptr(`{"city":"Paris"}`)}},
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCallOutput), ResponsesToolMessage: &schemas.ResponsesToolMessage{CallID: schemas.Ptr("call_1"), Output: &schemas.ResponsesToolMessageOutputStruct{ResponsesToolCallOutputStr: schemas.Ptr(`{"temp":21}`)}}},
		},
		Params: &schemas.ResponsesParameters{
			PreviousResponseID: schemas.Ptr("v1_prev"),
			Tools:              []schemas.ResponsesTool{{Type: schemas.ResponsesToolTypeFunction, Name: schemas.Ptr("get_weather"), ResponsesToolFunction: &schemas.ResponsesToolFunction{Parameters: &schemas.ToolFunctionParameters{Type: "object"}}}},
		},
	}
	b.ReportAllocs()
	for b.Loop() {
		native, err := ToGeminiInteractionsRequest(req)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := providerUtils.MarshalSorted(native); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGeminiInteractionsStreamEvent measures parsing and converting one text delta event, the
// per-chunk cost of an Interactions stream.
func BenchmarkGeminiInteractionsStreamEvent(b *testing.B) {
	data := []byte(`{"event_type":"step.delta","index":0,"delta":{"type":"text","text":"Hello there"}}`)
	state := NewGeminiInteractionsStreamState("gemini-3.8-flash")
	start, _ := ParseGeminiInteractionsEvent([]byte(`{"event_type":"step.start","index":0,"step":{"type":"model_output"}}`))
	_, _ = state.ToBifrostResponsesStream(start)
	b.ReportAllocs()
	for b.Loop() {
		event, err := ParseGeminiInteractionsEvent(data)
		if err != nil {
			b.Fatal(err)
		}
		_, _ = state.ToBifrostResponsesStream(event)
	}
}

// BenchmarkShouldUseGeminiInteractions measures the gate every Gemini and Vertex Responses request
// now passes, for the common case that stays on generateContent.
func BenchmarkShouldUseGeminiInteractions(b *testing.B) {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	req := &schemas.BifrostResponsesRequest{Model: "gemini-3.8-flash", Params: &schemas.ResponsesParameters{MaxOutputTokens: schemas.Ptr(100)}}
	b.ReportAllocs()
	for b.Loop() {
		if ShouldUseGeminiInteractions(ctx, req) {
			b.Fatal("plain request left generateContent")
		}
	}
}

// TestGeminiInteractionServerToolStepsToResponses pins the built-in tool steps the Interactions
// API documents onto their Responses items: a search is a web_search_call, code execution a
// code_interpreter_call, URL context one web_fetch_call per URL, a remote MCP call an mcp_call and
// file search a file_search_call. Call and result share one item, keyed by the call id. Steps
// with no Responses item (Google Maps) stay raw in provider_extra_fields. Citations on the text
// become annotations, and spoken output is an audio block, as on generateContent.
func TestGeminiInteractionServerToolStepsToResponses(t *testing.T) {
	body := `{
		"id":"v1_tools","object":"interaction","model":"gemini-3-flash-preview","status":"completed",
		"steps":[
			{"type":"google_search_call","id":"call_s","arguments":{"queries":["q1","q2"]},"search_type":"web_search","signature":"sg"},
			{"type":"google_search_result","call_id":"call_s","result":[{"search_suggestions":"<div/>"}],"is_error":false},
			{"type":"code_execution_call","id":"call_c","arguments":{"language":"python","code":"print(1)"}},
			{"type":"code_execution_result","call_id":"call_c","result":"1\n","is_error":false},
			{"type":"url_context_call","id":"call_u","arguments":{"urls":["https://a.example","https://b.example"]}},
			{"type":"url_context_result","call_id":"call_u","result":[{"url":"https://a.example","status":"success"},{"url":"https://b.example","status":"error"}]},
			{"type":"mcp_server_tool_call","id":"call_m","name":"roll","server_name":"dice","arguments":{"sides":6}},
			{"type":"mcp_server_tool_result","call_id":"call_m","name":"roll","server_name":"dice","result":[{"type":"text","text":"4"}]},
			{"type":"file_search_call","id":"call_f"},
			{"type":"file_search_result","call_id":"call_f"},
			{"type":"google_maps_call","id":"call_g","arguments":{}},
			{"type":"google_maps_result","call_id":"call_g","result":[]},
			{"type":"model_output","content":[
				{"type":"text","text":"Answer","annotations":[
					{"type":"url_citation","url":"https://a.example","title":"a","start_index":0,"end_index":6},
					{"type":"file_citation","document_uri":"fileSearchStores/s/documents/d","file_name":"d.pdf","start_index":0,"end_index":6}
				]},
				{"type":"audio","data":"QUJD","mime_type":"audio/wav"}
			]}
		]
	}`
	var interaction GeminiInteraction
	require.NoError(t, sonic.Unmarshal([]byte(body), &interaction))
	resp := interaction.ToBifrostResponsesResponse()

	var types []schemas.ResponsesMessageType
	for _, item := range resp.Output {
		types = append(types, *item.Type)
	}
	require.Equal(t, []schemas.ResponsesMessageType{
		schemas.ResponsesMessageTypeWebSearchCall,
		schemas.ResponsesMessageTypeCodeInterpreterCall,
		schemas.ResponsesMessageTypeWebFetchCall, schemas.ResponsesMessageTypeWebFetchCall,
		schemas.ResponsesMessageTypeMCPCall,
		schemas.ResponsesMessageTypeFileSearchCall,
		schemas.ResponsesMessageTypeMessage,
	}, types)

	search := resp.Output[0]
	assert.Equal(t, "call_s", *search.ID)
	assert.Equal(t, "completed", *search.Status)
	action := search.ResponsesToolMessage.Action.ResponsesWebSearchToolCallAction
	require.NotNil(t, action)
	assert.Equal(t, "search", action.Type)
	assert.Equal(t, []string{"q1", "q2"}, action.Queries)
	assert.Equal(t, "q1", *action.Query)

	code := resp.Output[1]
	assert.Equal(t, "call_c", *code.ID)
	assert.Equal(t, "completed", *code.Status)
	assert.Equal(t, "print(1)", *code.ResponsesToolMessage.ResponsesCodeInterpreterToolCall.Code)
	require.Len(t, code.ResponsesToolMessage.ResponsesCodeInterpreterToolCall.Outputs, 1)
	assert.Equal(t, "1\n", code.ResponsesToolMessage.ResponsesCodeInterpreterToolCall.Outputs[0].ResponsesCodeInterpreterOutputLogs.Logs)

	fetchA, fetchB := resp.Output[2], resp.Output[3]
	assert.Equal(t, "https://a.example", fetchA.ResponsesToolMessage.Action.ResponsesWebFetchToolCallAction.URL)
	assert.Equal(t, "completed", *fetchA.Status)
	assert.Equal(t, "https://b.example", fetchB.ResponsesToolMessage.Action.ResponsesWebFetchToolCallAction.URL)
	assert.Equal(t, "failed", *fetchB.Status, "a URL the tool could not read is a failed fetch")
	assert.NotEqual(t, *fetchA.ID, *fetchB.ID, "each URL is its own item")

	mcp := resp.Output[4]
	assert.Equal(t, "call_m", *mcp.ID)
	assert.Equal(t, "roll", *mcp.ResponsesToolMessage.Name)
	assert.Equal(t, "dice", *mcp.ResponsesToolMessage.ServerLabel)
	assert.JSONEq(t, `{"sides":6}`, *mcp.ResponsesToolMessage.Arguments)
	require.NotNil(t, mcp.ResponsesToolMessage.Output)
	assert.Equal(t, "4", *mcp.ResponsesToolMessage.Output.ResponsesToolCallOutputStr)
	assert.Equal(t, "completed", *mcp.Status)

	assert.Equal(t, "call_f", *resp.Output[5].ID)

	msg := resp.Output[6]
	require.Len(t, msg.Content.ContentBlocks, 2)
	text := msg.Content.ContentBlocks[0]
	require.Len(t, text.ResponsesOutputMessageContentText.Annotations, 2)
	url := text.ResponsesOutputMessageContentText.Annotations[0]
	assert.Equal(t, "url_citation", url.Type)
	assert.Equal(t, "https://a.example", *url.URL)
	assert.Equal(t, "a", *url.Title)
	assert.Equal(t, 0, *url.StartIndex)
	assert.Equal(t, 6, *url.EndIndex)
	file := text.ResponsesOutputMessageContentText.Annotations[1]
	assert.Equal(t, "file_citation", file.Type)
	assert.Equal(t, "d.pdf", *file.Filename)
	assert.Equal(t, "fileSearchStores/s/documents/d", *file.FileID)
	audio := msg.Content.ContentBlocks[1]
	assert.Equal(t, schemas.ResponsesInputMessageContentBlockTypeAudio, audio.Type)
	require.NotNil(t, audio.Audio)
	assert.Equal(t, "wav", audio.Audio.Format)
	assert.Equal(t, "QUJD", audio.Audio.Data)

	steps, ok := resp.ProviderExtraFields["steps"].([]GeminiInteractionsItem)
	require.True(t, ok, "steps with no Responses item stay raw")
	require.Len(t, steps, 2)
	assert.Equal(t, "google_maps_call", steps[0].Type)
	assert.Equal(t, "google_maps_result", steps[1].Type)
}

// TestGeminiInteractionsUsageCountsToolUseAndSearches pins usage the way Google bills it: tool
// use prompt tokens (what a code run or fetched page feeds back to the model) are input, so input
// and output add up to total_tokens, and grounding_tool_count's search count is the per-query
// search fee on Gemini 3 and later, one per request before that. Numbers are from a live
// code_execution run.
func TestGeminiInteractionsUsageCountsToolUseAndSearches(t *testing.T) {
	usage := &GeminiInteractionsUsage{
		TotalTokens: 1471, TotalInputTokens: 339, TotalOutputTokens: 218, TotalToolUseTokens: 445, TotalThoughtTokens: 469,
		GroundingToolCount: []GeminiInteractionsGroundingToolCount{{Type: "google_search", Count: 3}, {Type: "google_maps", Count: 1}},
	}
	got := usage.toResponsesUsage("gemini-3-flash-preview")
	assert.Equal(t, 784, got.InputTokens, "tool use prompt tokens are input")
	assert.Equal(t, 687, got.OutputTokens)
	assert.Equal(t, got.TotalTokens, got.InputTokens+got.OutputTokens)
	require.NotNil(t, got.ToolUsage)
	assert.Equal(t, 3, got.ToolUsage.WebSearch.NumRequests)

	older := usage.toResponsesUsage("gemini-2.5-flash")
	require.NotNil(t, older.ToolUsage)
	assert.Equal(t, 1, older.ToolUsage.WebSearch.NumRequests, "before Gemini 3 a grounded request is one search unit")

	plain := (&GeminiInteractionsUsage{TotalTokens: 3, TotalInputTokens: 1, TotalOutputTokens: 2}).toResponsesUsage("gemini-3-flash-preview")
	assert.Nil(t, plain.ToolUsage)
}

// TestGeminiInteractionRequiresContinuation pins a run Google paused for another request: it is
// incomplete, and the continuation_token the next request must send is surfaced.
func TestGeminiInteractionRequiresContinuation(t *testing.T) {
	var interaction GeminiInteraction
	require.NoError(t, sonic.Unmarshal([]byte(`{"id":"v1_long","status":"requires_continuation","continuation_token":"tok_1","model":"gemini-3-flash-preview","steps":[{"type":"model_output","content":[{"type":"text","text":"Part one"}]}]}`), &interaction))
	resp := interaction.ToBifrostResponsesResponse()
	assert.Equal(t, schemas.ResponsesResponseStatusIncomplete, *resp.Status)
	require.NotNil(t, resp.IncompleteDetails)
	assert.Equal(t, GeminiInteractionStatusRequiresContinuation, resp.IncompleteDetails.Reason)
	assert.Equal(t, "tok_1", resp.ProviderExtraFields["continuation_token"])

	state := NewGeminiInteractionsStreamState("gemini-3-flash-preview")
	var last *schemas.BifrostResponsesStreamResponse
	for _, event := range interactionsEvents(t,
		`{"event_type":"interaction.created","interaction":{"id":"v1_long","status":"in_progress"}}`,
		`{"event_type":"interaction.status_update","interaction_id":"v1_long","status":"requires_continuation","continuation_token":"tok_2"}`,
		`{"event_type":"interaction.completed","interaction":{"id":"v1_long","status":"requires_continuation","usage":{"total_tokens":3,"total_input_tokens":1,"total_output_tokens":2}}}`,
	) {
		out, err := state.ToBifrostResponsesStream(event)
		require.Nil(t, err)
		if len(out) > 0 {
			last = out[len(out)-1]
		}
	}
	assert.Equal(t, schemas.ResponsesStreamResponseTypeIncomplete, last.Type)
	assert.Equal(t, "tok_2", last.Response.ProviderExtraFields["continuation_token"], "the stream's latest token is the one to resume from")
}

// TestGeminiInteractionsStreamServerToolsAndCitations replays a live google_search stream (store
// false, trimmed). Built-in steps stream their whole call or result in one delta of their own
// type, citations arrive as text_annotation_delta, and interaction.completed carries usage but no
// steps, so the final response is built from the items the stream produced.
func TestGeminiInteractionsStreamServerToolsAndCitations(t *testing.T) {
	events := interactionsEvents(t,
		`{"interaction":{"object":"interaction","model":"gemini-3-flash-preview"},"event_type":"interaction.created"}`,
		`{"status":"in_progress","event_type":"interaction.status_update"}`,
		`{"index":0,"step":{"id":"call_1","signature":"","type":"google_search_call","search_type":"web_search"},"event_type":"step.start"}`,
		`{"index":0,"delta":{"signature":"s","type":"google_search_call","arguments":{"queries":["who won"]}},"event_type":"step.delta"}`,
		`{"index":0,"event_type":"step.stop"}`,
		`{"index":1,"step":{"call_id":"call_1","signature":"","type":"google_search_result"},"event_type":"step.start"}`,
		`{"index":1,"delta":{"signature":"s","type":"google_search_result","result":[{"search_suggestions":"<div/>"}],"is_error":false},"event_type":"step.delta"}`,
		`{"index":1,"event_type":"step.stop"}`,
		`{"index":2,"step":{"type":"thought"},"event_type":"step.start"}`,
		`{"index":2,"delta":{"signature":"sig","type":"thought_signature"},"event_type":"step.delta"}`,
		`{"index":2,"event_type":"step.stop"}`,
		`{"index":3,"step":{"type":"model_output"},"event_type":"step.start"}`,
		`{"index":3,"delta":{"text":"Kane won.","type":"text"},"event_type":"step.delta"}`,
		`{"index":3,"delta":{"annotations":[{"start_index":0,"end_index":9,"url":"https://r.example","title":"ballondor.com","type":"url_citation"}],"type":"text_annotation_delta"},"event_type":"step.delta"}`,
		`{"index":3,"event_type":"step.stop"}`,
		`{"interaction":{"status":"completed","model":"gemini-3-flash-preview","usage":{"total_tokens":2261,"total_input_tokens":985,"total_output_tokens":47,"total_tool_use_tokens":0,"total_thought_tokens":1229,"grounding_tool_count":[{"type":"google_search","count":3}]}},"event_type":"interaction.completed"}`,
	)
	state := NewGeminiInteractionsStreamState("gemini-3-flash-preview")
	var all []*schemas.BifrostResponsesStreamResponse
	for _, event := range events {
		out, err := state.ToBifrostResponsesStream(event)
		require.Nil(t, err)
		all = append(all, out...)
	}

	var searchDone, annotation *schemas.BifrostResponsesStreamResponse
	searchAdded := false
	for _, ev := range all {
		switch ev.Type {
		case schemas.ResponsesStreamResponseTypeOutputItemAdded:
			if *ev.Item.Type == schemas.ResponsesMessageTypeWebSearchCall {
				searchAdded = true
				assert.Equal(t, "in_progress", *ev.Item.Status)
			}
		case schemas.ResponsesStreamResponseTypeOutputItemDone:
			if *ev.Item.Type == schemas.ResponsesMessageTypeWebSearchCall {
				searchDone = ev
			}
		case schemas.ResponsesStreamResponseTypeOutputTextAnnotationAdded:
			annotation = ev
		}
	}
	require.True(t, searchAdded, "a search opens its item when it starts")
	require.NotNil(t, searchDone, "and closes it when its result arrives")
	assert.Equal(t, "call_1", *searchDone.Item.ID)
	assert.Equal(t, "completed", *searchDone.Item.Status)
	assert.Equal(t, []string{"who won"}, searchDone.Item.ResponsesToolMessage.Action.ResponsesWebSearchToolCallAction.Queries)
	require.NotNil(t, annotation, "a citation is streamed as an annotation")
	assert.Equal(t, "https://r.example", *annotation.Annotation.URL)

	final := all[len(all)-1]
	require.Equal(t, schemas.ResponsesStreamResponseTypeCompleted, final.Type)
	var finalTypes []schemas.ResponsesMessageType
	for _, item := range final.Response.Output {
		finalTypes = append(finalTypes, *item.Type)
	}
	assert.Equal(t, []schemas.ResponsesMessageType{schemas.ResponsesMessageTypeWebSearchCall, schemas.ResponsesMessageTypeReasoning, schemas.ResponsesMessageTypeMessage}, finalTypes,
		"interaction.completed has no steps, so the final output is what the stream produced")
	text := final.Response.Output[2].Content.ContentBlocks[0]
	assert.Equal(t, "Kane won.", *text.Text)
	require.Len(t, text.ResponsesOutputMessageContentText.Annotations, 1)
	require.NotNil(t, final.Response.Usage.ToolUsage)
	assert.Equal(t, 3, final.Response.Usage.ToolUsage.WebSearch.NumRequests)
}

// TestToGeminiInteractionsToolsComputerUse pins the computer use tool both ways: a Responses
// computer tool is Interactions computer_use with its environment, and back.
func TestToGeminiInteractionsToolsComputerUse(t *testing.T) {
	tools, err := toGeminiInteractionsTools([]schemas.ResponsesTool{{
		Type:                            schemas.ResponsesToolTypeComputerUsePreview,
		ResponsesToolComputerUsePreview: &schemas.ResponsesToolComputerUsePreview{Environment: "browser"},
	}})
	require.NoError(t, err)
	require.Len(t, tools, 1)
	assert.Equal(t, "computer_use", tools[0].Type)
	assert.Equal(t, "browser", *tools[0].Environment)

	back, err := geminiInteractionsToolsToResponses([]GeminiInteractionsTool{{Type: "computer_use", Environment: schemas.Ptr("desktop")}})
	require.NoError(t, err)
	require.Len(t, back, 1)
	assert.Equal(t, schemas.ResponsesToolTypeComputerUsePreview, back[0].Type)
	assert.Equal(t, "desktop", back[0].ResponsesToolComputerUsePreview.Environment)
}

// TestToGeminiInteractionsInputSkipsServerToolItems pins replay of a stateless turn: the built-in
// tool items a previous Interactions response returned ran on Google's side and carry no
// signature to replay, so they are dropped from the input rather than refusing the request.
func TestToGeminiInteractionsInputSkipsServerToolItems(t *testing.T) {
	assistant := schemas.ResponsesInputMessageRoleAssistant
	req := &schemas.BifrostResponsesRequest{Provider: schemas.Gemini, Model: "gemini-3-flash-preview", Input: []schemas.ResponsesMessage{
		interactionsUserText("Who won?"),
		{ID: schemas.Ptr("call_s"), Type: schemas.Ptr(schemas.ResponsesMessageTypeWebSearchCall), Status: schemas.Ptr("completed"), ResponsesToolMessage: &schemas.ResponsesToolMessage{Action: &schemas.ResponsesToolMessageActionStruct{ResponsesWebSearchToolCallAction: &schemas.ResponsesWebSearchToolCallAction{Type: "search", Queries: []string{"q"}}}}},
		{ID: schemas.Ptr("call_c"), Type: schemas.Ptr(schemas.ResponsesMessageTypeCodeInterpreterCall), ResponsesToolMessage: &schemas.ResponsesToolMessage{ResponsesCodeInterpreterToolCall: &schemas.ResponsesCodeInterpreterToolCall{Code: schemas.Ptr("1")}}},
		{Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage), Role: &assistant, Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("Kane.")}},
		interactionsUserText("Since when?"),
	}}
	out, err := ToGeminiInteractionsRequest(req)
	require.NoError(t, err)
	require.NotNil(t, out.Input)
	var types []string
	for _, step := range out.Input.Items {
		types = append(types, step.Type)
	}
	assert.Equal(t, []string{"user_input", "model_output", "user_input"}, types)
}

// TestToGeminiInteractionServerToolItems pins the translated direction: an Interactions caller
// whose request ran on another provider gets that provider's built-in tool items as the steps the
// Interactions API documents, its citations as annotations and its search count as
// grounding_tool_count.
func TestToGeminiInteractionServerToolItems(t *testing.T) {
	text := schemas.ResponsesMessageContentBlock{
		Type: schemas.ResponsesOutputMessageContentTypeText, Text: schemas.Ptr("Kane."),
		ResponsesOutputMessageContentText: &schemas.ResponsesOutputMessageContentText{Annotations: []schemas.ResponsesOutputMessageContentTextAnnotation{
			{Type: "url_citation", URL: schemas.Ptr("https://r.example"), Title: schemas.Ptr("r"), StartIndex: schemas.Ptr(0), EndIndex: schemas.Ptr(5)},
		}},
	}
	resp := &schemas.BifrostResponsesResponse{
		ID: schemas.Ptr("resp_t"), Model: "gpt-5", Status: schemas.Ptr("completed"),
		Output: []schemas.ResponsesMessage{
			{ID: schemas.Ptr("ws_1"), Type: schemas.Ptr(schemas.ResponsesMessageTypeWebSearchCall), Status: schemas.Ptr("completed"), ResponsesToolMessage: &schemas.ResponsesToolMessage{Action: &schemas.ResponsesToolMessageActionStruct{ResponsesWebSearchToolCallAction: &schemas.ResponsesWebSearchToolCallAction{Type: "search", Query: schemas.Ptr("who won")}}}},
			{ID: schemas.Ptr("ci_1"), Type: schemas.Ptr(schemas.ResponsesMessageTypeCodeInterpreterCall), Status: schemas.Ptr("completed"), ResponsesToolMessage: &schemas.ResponsesToolMessage{ResponsesCodeInterpreterToolCall: &schemas.ResponsesCodeInterpreterToolCall{
				Code: schemas.Ptr("print(2)"), Outputs: []schemas.ResponsesCodeInterpreterOutput{{ResponsesCodeInterpreterOutputLogs: &schemas.ResponsesCodeInterpreterOutputLogs{Type: "logs", Logs: "2\n"}}},
			}}},
			{ID: schemas.Ptr("mcp_1"), Type: schemas.Ptr(schemas.ResponsesMessageTypeMCPCall), Status: schemas.Ptr("completed"), ResponsesToolMessage: &schemas.ResponsesToolMessage{
				Name: schemas.Ptr("roll"), ServerLabel: schemas.Ptr("dice"), Arguments: schemas.Ptr(`{"sides":6}`),
				Output: &schemas.ResponsesToolMessageOutputStruct{ResponsesToolCallOutputStr: schemas.Ptr("4")},
			}},
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage), Role: schemas.Ptr(schemas.ResponsesInputMessageRoleAssistant), Content: &schemas.ResponsesMessageContent{ContentBlocks: []schemas.ResponsesMessageContentBlock{text}}},
		},
		Usage: &schemas.ResponsesResponseUsage{InputTokens: 3, OutputTokens: 4, TotalTokens: 7, OutputTokensDetails: &schemas.ResponsesResponseOutputTokens{NumSearchQueries: schemas.Ptr(2)}},
	}
	raw, err := providerUtils.MarshalSorted(ToGeminiInteraction(resp))
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"id":"resp_t","object":"interaction","model":"gpt-5","status":"completed",
		"steps":[
			{"type":"google_search_call","id":"ws_1","arguments":{"queries":["who won"]}},
			{"type":"google_search_result","call_id":"ws_1","result":[]},
			{"type":"code_execution_call","id":"ci_1","arguments":{"code":"print(2)","language":"python"}},
			{"type":"code_execution_result","call_id":"ci_1","result":"2\n"},
			{"type":"mcp_server_tool_call","id":"mcp_1","name":"roll","server_name":"dice","arguments":{"sides":6}},
			{"type":"mcp_server_tool_result","call_id":"mcp_1","name":"roll","server_name":"dice","result":"4"},
			{"type":"model_output","content":[{"type":"text","text":"Kane.","annotations":[{"type":"url_citation","url":"https://r.example","title":"r","start_index":0,"end_index":5}]}]}
		],
		"usage":{"total_input_tokens":3,"total_output_tokens":4,"total_tokens":7,"grounding_tool_count":[{"type":"google_search","count":2}]}
	}`, string(raw))
}

// TestBifrostToGeminiInteractionsStreamServerTools pins a translated stream that carries a
// built-in tool item: the finished item becomes its call and result steps, each with its own step
// index, and the steps around it keep distinct indexes.
func TestBifrostToGeminiInteractionsStreamServerTools(t *testing.T) {
	state := NewBifrostToGeminiInteractionsStreamState()
	zero, one := 0, 1
	search := &schemas.ResponsesMessage{ID: schemas.Ptr("ws_1"), Type: schemas.Ptr(schemas.ResponsesMessageTypeWebSearchCall), Status: schemas.Ptr("completed"), ResponsesToolMessage: &schemas.ResponsesToolMessage{Action: &schemas.ResponsesToolMessageActionStruct{ResponsesWebSearchToolCallAction: &schemas.ResponsesWebSearchToolCallAction{Type: "search", Queries: []string{"q"}}}}}
	var events []*GeminiInteractionsEvent
	for _, ev := range []*schemas.BifrostResponsesStreamResponse{
		{Type: schemas.ResponsesStreamResponseTypeCreated, Response: &schemas.BifrostResponsesResponse{ID: schemas.Ptr("resp_s"), Model: "gpt-5"}},
		{Type: schemas.ResponsesStreamResponseTypeOutputItemAdded, OutputIndex: &zero, Item: &schemas.ResponsesMessage{ID: schemas.Ptr("ws_1"), Type: schemas.Ptr(schemas.ResponsesMessageTypeWebSearchCall), Status: schemas.Ptr("in_progress")}},
		{Type: schemas.ResponsesStreamResponseTypeOutputItemDone, OutputIndex: &zero, Item: search},
		{Type: schemas.ResponsesStreamResponseTypeOutputItemAdded, OutputIndex: &one, Item: &schemas.ResponsesMessage{Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage), Role: schemas.Ptr(schemas.ResponsesInputMessageRoleAssistant)}},
		{Type: schemas.ResponsesStreamResponseTypeOutputTextDelta, OutputIndex: &one, Delta: schemas.Ptr("Hi")},
		{Type: schemas.ResponsesStreamResponseTypeOutputItemDone, OutputIndex: &one},
	} {
		events = append(events, state.ToGeminiInteractionsStreamEvents(ev)...)
	}
	var got []string
	for _, ev := range events {
		switch ev.EventType {
		case GeminiInteractionsEventStepStart:
			got = append(got, fmt.Sprintf("start %d %s", *ev.Index, ev.Step.Type))
		case GeminiInteractionsEventStepDelta:
			got = append(got, fmt.Sprintf("delta %d %s", *ev.Index, ev.Delta.Type))
		case GeminiInteractionsEventStepStop:
			got = append(got, fmt.Sprintf("stop %d", *ev.Index))
		}
	}
	assert.Equal(t, []string{
		"start 0 google_search_call", "stop 0",
		"start 1 google_search_result", "stop 1",
		"start 2 model_output", "delta 2 text", "stop 2",
	}, got)
}
