package vertex

import (
	"context"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestVertexInteractionsTarget pins where Vertex sends Interactions calls: the key's own project and
// region under v1beta1 (Vertex answers 404 on v1), global when the key names no region.
func TestVertexInteractionsTarget(t *testing.T) {
	cases := []struct {
		name, region, wantURL string
	}{
		{"no region is global", "", "https://aiplatform.googleapis.com/v1beta1/projects/proj-1/locations/global/interactions"},
		{"global region", "global", "https://aiplatform.googleapis.com/v1beta1/projects/proj-1/locations/global/interactions"},
		{"a regional key keeps its region", "us-central1", "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/proj-1/locations/us-central1/interactions"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.wantURL, vertexInteractionsCollectionURL(tc.region, "proj-1"))
		})
	}

	provider := &VertexProvider{}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	_, bifrostErr := provider.interactionsTarget(ctx, schemas.Key{VertexKeyConfig: &schemas.VertexKeyConfig{}})
	require.NotNil(t, bifrostErr, "a key with no project cannot address an interaction")
}

// TestUseVertexInteractions pins which Vertex Responses requests run on Interactions: a Gemini model
// or an agent that needs stored state, a background run, or came in on the Interactions route. A
// Claude model on Vertex keeps its Anthropic surface even when it names a previous response.
func TestUseVertexInteractions(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	withPrevious := func(model string) *schemas.BifrostResponsesRequest {
		return &schemas.BifrostResponsesRequest{Model: model, Params: &schemas.ResponsesParameters{PreviousResponseID: schemas.Ptr("v1_prev")}}
	}
	assert.True(t, useVertexInteractions(ctx, withPrevious("gemini-3.8-flash")))
	assert.True(t, useVertexInteractions(ctx, &schemas.BifrostResponsesRequest{Model: "deep-research-preview-04-2026"}))
	assert.False(t, useVertexInteractions(ctx, withPrevious("claude-sonnet-5-5")))
	assert.False(t, useVertexInteractions(ctx, &schemas.BifrostResponsesRequest{Model: "gemini-3.8-flash"}))

	routed := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	routed.SetValue(schemas.BifrostContextKeyInteractionsAPI, true)
	assert.True(t, useVertexInteractions(routed, &schemas.BifrostResponsesRequest{Model: "gemini-3.8-flash"}))
	assert.False(t, useVertexInteractions(routed, &schemas.BifrostResponsesRequest{Model: "claude-sonnet-5-5"}))
}
