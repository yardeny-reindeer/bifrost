package gemini

import (
	"bytes"
	"encoding/json"
	"strconv"
	"strings"

	"github.com/bytedance/sonic"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

// Built-in tool steps. Google runs these tools itself and reports each run as a call step and a
// result step that names the call by call_id. Each pair maps onto the Responses item for the same
// tool, so a Responses caller sees a search, a code run, a fetched page or a remote MCP call the
// way it would from any other provider. Steps with no Responses item (google_maps, processing)
// stay raw under provider_extra_fields.steps.
const (
	GeminiInteractionsStepGoogleSearchCall    = "google_search_call"
	GeminiInteractionsStepGoogleSearchResult  = "google_search_result"
	GeminiInteractionsStepCodeExecutionCall   = "code_execution_call"
	GeminiInteractionsStepCodeExecutionResult = "code_execution_result"
	GeminiInteractionsStepURLContextCall      = "url_context_call"
	GeminiInteractionsStepURLContextResult    = "url_context_result"
	GeminiInteractionsStepFileSearchCall      = "file_search_call"
	GeminiInteractionsStepFileSearchResult    = "file_search_result"
	GeminiInteractionsStepMCPCall             = "mcp_server_tool_call"
	GeminiInteractionsStepMCPResult           = "mcp_server_tool_result"
)

// geminiInteractionsDeltaAnnotations carries citations for the text of a model_output step.
const geminiInteractionsDeltaAnnotations = "text_annotation_delta"

// geminiInteractionsGoogleSearch is the grounding_tool_count type of a Google Search run.
const geminiInteractionsGoogleSearch = "google_search"

// geminiInteractionsToolArguments is the arguments object of a built-in call step.
type geminiInteractionsToolArguments struct {
	Queries  []string `json:"queries,omitempty"`  // google_search_call, file_search_call
	Code     string   `json:"code,omitempty"`     // code_execution_call
	Language string   `json:"language,omitempty"` // code_execution_call
	URLs     []string `json:"urls,omitempty"`     // url_context_call
}

// geminiInteractionsURLResult is one entry of a url_context_result.
type geminiInteractionsURLResult struct {
	URL    string `json:"url"`
	Status string `json:"status"`
}

// geminiInteractionsSearchResult is the part of a google_search_result entry Bifrost reads.
type geminiInteractionsSearchResult struct {
	URL   string `json:"url,omitempty"`
	Title string `json:"title,omitempty"`
}

// geminiInteractionsAnnotation is a citation on a text block: url_citation or file_citation.
// The other documented kinds (place_citation, speech and word annotations) have no Responses
// annotation and are not mapped.
type geminiInteractionsAnnotation struct {
	Type        string  `json:"type"`
	URL         *string `json:"url,omitempty"`
	Title       *string `json:"title,omitempty"`
	DocumentURI *string `json:"document_uri,omitempty"`
	FileName    *string `json:"file_name,omitempty"`
	StartIndex  *int    `json:"start_index,omitempty"`
	EndIndex    *int    `json:"end_index,omitempty"`
}

func isGeminiServerToolCall(stepType string) bool {
	switch stepType {
	case GeminiInteractionsStepGoogleSearchCall, GeminiInteractionsStepCodeExecutionCall, GeminiInteractionsStepURLContextCall,
		GeminiInteractionsStepFileSearchCall, GeminiInteractionsStepMCPCall:
		return true
	}
	return false
}

func isGeminiServerToolResult(stepType string) bool {
	switch stepType {
	case GeminiInteractionsStepGoogleSearchResult, GeminiInteractionsStepCodeExecutionResult, GeminiInteractionsStepURLContextResult,
		GeminiInteractionsStepFileSearchResult, GeminiInteractionsStepMCPResult:
		return true
	}
	return false
}

// isResponsesServerToolItem reports whether a Responses item is a tool the provider ran itself.
func isResponsesServerToolItem(itemType schemas.ResponsesMessageType) bool {
	switch itemType {
	case schemas.ResponsesMessageTypeWebSearchCall, schemas.ResponsesMessageTypeCodeInterpreterCall, schemas.ResponsesMessageTypeWebFetchCall,
		schemas.ResponsesMessageTypeFileSearchCall, schemas.ResponsesMessageTypeMCPCall:
		return true
	}
	return false
}

// geminiServerToolCallToResponses renders a built-in call step as its Responses items, in
// progress until the result step arrives. url_context reads several URLs in one call; each is
// its own web_fetch_call, as a Responses web fetch names one URL.
func geminiServerToolCallToResponses(step GeminiInteractionsItem) []schemas.ResponsesMessage {
	id := ""
	if step.ID != nil {
		id = *step.ID
	}
	var args geminiInteractionsToolArguments
	if len(step.Arguments) > 0 {
		_ = sonic.Unmarshal(step.Arguments, &args)
	}
	item := func(itemType schemas.ResponsesMessageType, itemID string, tool *schemas.ResponsesToolMessage) schemas.ResponsesMessage {
		return schemas.ResponsesMessage{ID: schemas.Ptr(itemID), Type: schemas.Ptr(itemType), Status: schemas.Ptr("in_progress"), ResponsesToolMessage: tool}
	}
	switch step.Type {
	case GeminiInteractionsStepGoogleSearchCall:
		action := &schemas.ResponsesWebSearchToolCallAction{Type: "search", Queries: args.Queries}
		if len(args.Queries) > 0 {
			action.Query = schemas.Ptr(args.Queries[0])
		}
		return []schemas.ResponsesMessage{item(schemas.ResponsesMessageTypeWebSearchCall, id, &schemas.ResponsesToolMessage{
			Action: &schemas.ResponsesToolMessageActionStruct{ResponsesWebSearchToolCallAction: action},
		})}
	case GeminiInteractionsStepCodeExecutionCall:
		code := args.Code
		return []schemas.ResponsesMessage{item(schemas.ResponsesMessageTypeCodeInterpreterCall, id, &schemas.ResponsesToolMessage{
			CallID:                           schemas.Ptr(id),
			ResponsesCodeInterpreterToolCall: &schemas.ResponsesCodeInterpreterToolCall{Code: &code, Outputs: []schemas.ResponsesCodeInterpreterOutput{}},
		})}
	case GeminiInteractionsStepURLContextCall:
		urls := args.URLs
		if len(urls) == 0 {
			urls = []string{""}
		}
		items := make([]schemas.ResponsesMessage, 0, len(urls))
		for i, u := range urls {
			itemID := id
			if len(urls) > 1 {
				itemID = id + "_" + strconv.Itoa(i)
			}
			items = append(items, item(schemas.ResponsesMessageTypeWebFetchCall, itemID, &schemas.ResponsesToolMessage{
				Action: &schemas.ResponsesToolMessageActionStruct{ResponsesWebFetchToolCallAction: &schemas.ResponsesWebFetchToolCallAction{Type: "fetch", URL: u}},
			}))
		}
		return items
	case GeminiInteractionsStepFileSearchCall:
		queries := args.Queries
		if queries == nil {
			queries = []string{}
		}
		return []schemas.ResponsesMessage{item(schemas.ResponsesMessageTypeFileSearchCall, id, &schemas.ResponsesToolMessage{
			ResponsesFileSearchToolCall: &schemas.ResponsesFileSearchToolCall{Queries: queries},
		})}
	case GeminiInteractionsStepMCPCall:
		arguments := "{}"
		if len(step.Arguments) > 0 && !bytes.Equal(step.Arguments, []byte("null")) {
			arguments = string(step.Arguments)
		}
		return []schemas.ResponsesMessage{item(schemas.ResponsesMessageTypeMCPCall, id, &schemas.ResponsesToolMessage{
			Name: step.Name, ServerLabel: step.ServerName, Arguments: &arguments,
		})}
	}
	return nil
}

// applyGeminiServerToolResult folds a result step onto the items its call produced.
func applyGeminiServerToolResult(items []*schemas.ResponsesMessage, step GeminiInteractionsItem) {
	status := "completed"
	if step.IsError != nil && *step.IsError {
		status = "failed"
	}
	for _, it := range items {
		itemStatus := status
		tool := it.ResponsesToolMessage
		switch step.Type {
		case GeminiInteractionsStepGoogleSearchResult:
			var results []geminiInteractionsSearchResult
			if len(step.Result) > 0 && sonic.Unmarshal(step.Result, &results) == nil && tool != nil && tool.Action != nil && tool.Action.ResponsesWebSearchToolCallAction != nil {
				for _, r := range results {
					if r.URL == "" {
						continue
					}
					source := schemas.ResponsesWebSearchToolCallActionSearchSource{Type: "url", URL: r.URL}
					if r.Title != "" {
						source.Title = schemas.Ptr(r.Title)
					}
					tool.Action.ResponsesWebSearchToolCallAction.Sources = append(tool.Action.ResponsesWebSearchToolCallAction.Sources, source)
				}
			}
		case GeminiInteractionsStepCodeExecutionResult:
			if tool != nil && tool.ResponsesCodeInterpreterToolCall != nil {
				logs := geminiInteractionsResultText(step.Result)
				tool.ResponsesCodeInterpreterToolCall.Outputs = append(tool.ResponsesCodeInterpreterToolCall.Outputs, schemas.ResponsesCodeInterpreterOutput{
					ResponsesCodeInterpreterOutputLogs: &schemas.ResponsesCodeInterpreterOutputLogs{Type: "logs", Logs: logs},
				})
			}
		case GeminiInteractionsStepURLContextResult:
			var results []geminiInteractionsURLResult
			if len(step.Result) > 0 && sonic.Unmarshal(step.Result, &results) == nil && tool != nil && tool.Action != nil && tool.Action.ResponsesWebFetchToolCallAction != nil {
				for _, r := range results {
					if r.URL == tool.Action.ResponsesWebFetchToolCallAction.URL && r.Status != "success" {
						itemStatus = "failed"
					}
				}
			}
		case GeminiInteractionsStepMCPResult:
			if tool != nil {
				output := geminiInteractionsResultText(step.Result)
				tool.Output = &schemas.ResponsesToolMessageOutputStruct{ResponsesToolCallOutputStr: &output}
			}
		}
		it.Status = &itemStatus
	}
}

// geminiInteractionsResultText reads a built-in tool result as text: the string itself, the
// text of a content list, or the raw JSON of anything else.
func geminiInteractionsResultText(raw json.RawMessage) string {
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return ""
	}
	var text string
	if sonic.Unmarshal(raw, &text) == nil {
		return text
	}
	var blocks []GeminiInteractionsItem
	if sonic.Unmarshal(raw, &blocks) == nil {
		var b strings.Builder
		for _, block := range blocks {
			if block.Text != nil {
				b.WriteString(*block.Text)
			}
		}
		return b.String()
	}
	return string(raw)
}

// geminiInteractionsAnnotationsToResponses maps the citations on a text block.
func geminiInteractionsAnnotationsToResponses(raw json.RawMessage) []schemas.ResponsesOutputMessageContentTextAnnotation {
	out := []schemas.ResponsesOutputMessageContentTextAnnotation{}
	if len(raw) == 0 {
		return out
	}
	var annotations []geminiInteractionsAnnotation
	if sonic.Unmarshal(raw, &annotations) != nil {
		return out
	}
	for _, a := range annotations {
		switch a.Type {
		case "url_citation":
			out = append(out, schemas.ResponsesOutputMessageContentTextAnnotation{Type: "url_citation", URL: a.URL, Title: a.Title, StartIndex: a.StartIndex, EndIndex: a.EndIndex})
		case "file_citation":
			out = append(out, schemas.ResponsesOutputMessageContentTextAnnotation{Type: "file_citation", FileID: a.DocumentURI, Filename: a.FileName, StartIndex: a.StartIndex, EndIndex: a.EndIndex})
		}
	}
	return out
}

// responsesAnnotationsToGeminiInteractions maps Responses citations back to Interactions ones.
func responsesAnnotationsToGeminiInteractions(annotations []schemas.ResponsesOutputMessageContentTextAnnotation) json.RawMessage {
	out := make([]geminiInteractionsAnnotation, 0, len(annotations))
	for _, a := range annotations {
		switch a.Type {
		case "url_citation":
			out = append(out, geminiInteractionsAnnotation{Type: "url_citation", URL: a.URL, Title: a.Title, StartIndex: a.StartIndex, EndIndex: a.EndIndex})
		case "file_citation":
			out = append(out, geminiInteractionsAnnotation{Type: "file_citation", DocumentURI: a.FileID, FileName: a.Filename, StartIndex: a.StartIndex, EndIndex: a.EndIndex})
		}
	}
	if len(out) == 0 {
		return nil
	}
	raw, err := providerUtils.MarshalSorted(out)
	if err != nil {
		return nil
	}
	return raw
}

// geminiInteractionsSearchUnits is the billable Google Search units of a run: one per query on
// Gemini 3 and later, one per grounded request before that, the same rule generateContent
// follows. Other grounding tools (google_maps) are not search units.
func geminiInteractionsSearchUnits(counts []GeminiInteractionsGroundingToolCount, model string) int {
	total := 0
	for _, c := range counts {
		if c.Type == geminiInteractionsGoogleSearch {
			total += c.Count
		}
	}
	if total <= 0 {
		return 0
	}
	if !isGemini3Plus(model) {
		return 1
	}
	return total
}

// responsesServerToolItemToGeminiInteractionSteps renders a built-in tool item another provider
// ran as the call and result steps the Interactions API documents for that tool.
func responsesServerToolItemToGeminiInteractionSteps(item schemas.ResponsesMessage) []GeminiInteractionsItem {
	tool := item.ResponsesToolMessage
	if tool == nil || item.Type == nil {
		return nil
	}
	id := ""
	switch {
	case item.ID != nil:
		id = *item.ID
	case tool.CallID != nil:
		id = *tool.CallID
	}
	var isError *bool
	if item.Status != nil && *item.Status == "failed" {
		isError = schemas.Ptr(true)
	}
	marshal := func(v interface{}) json.RawMessage {
		raw, err := providerUtils.MarshalSorted(v)
		if err != nil {
			return nil
		}
		return raw
	}
	call := GeminiInteractionsItem{ID: schemas.Ptr(id)}
	result := GeminiInteractionsItem{CallID: schemas.Ptr(id), IsError: isError}
	switch *item.Type {
	case schemas.ResponsesMessageTypeWebSearchCall:
		var args geminiInteractionsToolArguments
		if tool.Action != nil && tool.Action.ResponsesWebSearchToolCallAction != nil {
			action := tool.Action.ResponsesWebSearchToolCallAction
			args.Queries = action.Queries
			if len(args.Queries) == 0 && action.Query != nil {
				args.Queries = []string{*action.Query}
			}
		}
		call.Type, call.Arguments = GeminiInteractionsStepGoogleSearchCall, marshal(args)
		result.Type, result.Result = GeminiInteractionsStepGoogleSearchResult, json.RawMessage("[]")
	case schemas.ResponsesMessageTypeCodeInterpreterCall:
		args := geminiInteractionsToolArguments{Language: "python"}
		var logs strings.Builder
		if ci := tool.ResponsesCodeInterpreterToolCall; ci != nil {
			if ci.Code != nil {
				args.Code = *ci.Code
			}
			for _, o := range ci.Outputs {
				if o.ResponsesCodeInterpreterOutputLogs != nil {
					logs.WriteString(o.ResponsesCodeInterpreterOutputLogs.Logs)
				}
			}
		}
		call.Type, call.Arguments = GeminiInteractionsStepCodeExecutionCall, marshal(args)
		result.Type, result.Result = GeminiInteractionsStepCodeExecutionResult, marshal(logs.String())
	case schemas.ResponsesMessageTypeWebFetchCall:
		url := ""
		if tool.Action != nil && tool.Action.ResponsesWebFetchToolCallAction != nil {
			url = tool.Action.ResponsesWebFetchToolCallAction.URL
		}
		status := "success"
		if isError != nil {
			status, isError, result.IsError = "error", nil, nil
		}
		call.Type, call.Arguments = GeminiInteractionsStepURLContextCall, marshal(geminiInteractionsToolArguments{URLs: []string{url}})
		result.Type, result.Result = GeminiInteractionsStepURLContextResult, marshal([]geminiInteractionsURLResult{{URL: url, Status: status}})
	case schemas.ResponsesMessageTypeFileSearchCall:
		call.Type = GeminiInteractionsStepFileSearchCall
		result.Type = GeminiInteractionsStepFileSearchResult
	case schemas.ResponsesMessageTypeMCPCall:
		args := json.RawMessage("{}")
		if tool.Arguments != nil && json.Valid([]byte(*tool.Arguments)) {
			args = json.RawMessage(*tool.Arguments)
		}
		call.Type, call.Name, call.ServerName, call.Arguments = GeminiInteractionsStepMCPCall, tool.Name, tool.ServerLabel, args
		if tool.Output == nil {
			return []GeminiInteractionsItem{call}
		}
		output := ""
		if tool.Output.ResponsesToolCallOutputStr != nil {
			output = *tool.Output.ResponsesToolCallOutputStr
		}
		result.Type, result.Name, result.ServerName, result.Result = GeminiInteractionsStepMCPResult, tool.Name, tool.ServerLabel, marshal(output)
	default:
		return nil
	}
	return []GeminiInteractionsItem{call, result}
}

// geminiInteractionsComputerEnvironment maps a Responses computer environment onto the three
// the Interactions computer_use tool takes. Desktop operating systems are "desktop"; anything
// unset or unknown is "browser", the tool's own default.
func geminiInteractionsComputerEnvironment(environment string) string {
	switch strings.ToLower(environment) {
	case "mobile":
		return "mobile"
	case "desktop", "mac", "windows", "ubuntu", "linux":
		return "desktop"
	}
	return "browser"
}
