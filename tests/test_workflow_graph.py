"""
Unit tests for the LangGraph workflow structure.

Validates that the state machine compiles correctly, contains the expected
nodes, and that the GraphState TypedDict has the correct schema.
These tests mock all external dependencies (LLM, embeddings, ChromaDB).
"""

from unittest.mock import patch, MagicMock

import pytest

from src.workflows.workflow_nodes import GraphState


# --------------------------------------------------------------------- #
# GraphState schema
# --------------------------------------------------------------------- #
class TestGraphState:
    def test_graph_state_has_required_keys(self):
        """GraphState TypedDict should declare all expected keys."""
        expected_keys = {"query", "documents", "risk_analysis", "final_answer", "overall_report"}
        assert expected_keys == set(GraphState.__annotations__.keys())

    def test_graph_state_is_instantiable(self):
        """A valid GraphState dict should match the expected shape."""
        state: GraphState = {
            "query": "What are the risks?",
            "documents": [],
            "risk_analysis": [],
            "final_answer": "",
            "overall_report": {},
        }
        assert state["query"] == "What are the risks?"
        assert isinstance(state["documents"], list)


# --------------------------------------------------------------------- #
# Workflow compilation and structure
# --------------------------------------------------------------------- #
class TestWorkflowGraph:
    @patch("src.workflows.workflow_nodes.VectorStoreManager")
    @patch("src.workflows.workflow_nodes.HuggingFaceEndpoint")
    @patch("src.workflows.workflow_nodes.ChatHuggingFace")
    @patch("src.workflows.workflow_nodes.RiskScorer")
    def test_workflow_compiles(self, mock_scorer, mock_chat, mock_endpoint, mock_vs):
        """create_workflow() should return a compiled graph without errors."""
        mock_endpoint.return_value = MagicMock()
        mock_chat.return_value = MagicMock()
        mock_vs.return_value = MagicMock()
        mock_scorer.return_value = MagicMock()

        from src.workflows.workflow_graph import create_workflow

        workflow = create_workflow()
        # A compiled LangGraph has an `invoke` method.
        assert hasattr(workflow, "invoke") or hasattr(workflow, "ainvoke")

    @patch("src.workflows.workflow_nodes.VectorStoreManager")
    @patch("src.workflows.workflow_nodes.HuggingFaceEndpoint")
    @patch("src.workflows.workflow_nodes.ChatHuggingFace")
    @patch("src.workflows.workflow_nodes.RiskScorer")
    def test_workflow_has_expected_nodes(self, mock_scorer, mock_chat, mock_endpoint, mock_vs):
        """The compiled graph should contain retrieve, analyze_risk, generate_answer."""
        mock_endpoint.return_value = MagicMock()
        mock_chat.return_value = MagicMock()
        mock_vs.return_value = MagicMock()
        mock_scorer.return_value = MagicMock()

        from src.workflows.workflow_graph import create_workflow

        workflow = create_workflow()

        # LangGraph's get_graph() returns a graph whose .nodes can be
        # either strings or objects depending on the version installed.
        graph = workflow.get_graph()
        node_ids = set()
        for node in graph.nodes:
            # Some versions return node objects with .id, others return strings.
            if isinstance(node, str):
                node_ids.add(node)
            else:
                node_ids.add(node.id)

        assert "retrieve" in node_ids
        assert "analyze_risk" in node_ids
        assert "generate_answer" in node_ids

    @patch("src.workflows.workflow_nodes.VectorStoreManager")
    @patch("src.workflows.workflow_nodes.HuggingFaceEndpoint")
    @patch("src.workflows.workflow_nodes.ChatHuggingFace")
    @patch("src.workflows.workflow_nodes.RiskScorer")
    def test_workflow_entry_point(self, mock_scorer, mock_chat, mock_endpoint, mock_vs):
        """The workflow should start at the 'retrieve' node."""
        mock_endpoint.return_value = MagicMock()
        mock_chat.return_value = MagicMock()
        mock_vs.return_value = MagicMock()
        mock_scorer.return_value = MagicMock()

        from src.workflows.workflow_graph import create_workflow

        workflow = create_workflow()
        graph = workflow.get_graph()

        # Check that __start__ is present in the graph
        node_ids = set()
        for node in graph.nodes:
            nid = node if isinstance(node, str) else node.id
            node_ids.add(nid)

        assert "__start__" in node_ids

        # Find edges from __start__ — edges can be tuples or objects.
        target_ids = set()
        for edge in graph.edges:
            if isinstance(edge, tuple):
                src, tgt = edge[0], edge[1]
            else:
                src, tgt = edge.source, edge.target
            if src == "__start__":
                target_ids.add(tgt)

        assert "retrieve" in target_ids
