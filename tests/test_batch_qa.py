import asyncio
from unittest.mock import MagicMock
from fastapi import Request

from app.api import routes
from app.models.schemas import BatchQuestionRequest, BatchQuestionItem, AnswerResponse


def test_batch_qa_endpoint():
    fake_qa_chain = MagicMock()
    fake_qa_chain.ask.return_value = AnswerResponse(
        answer="هذا جواب تجريبي",
        confidence=0.92,
        sources=[],
    )

    fake_vector_store = MagicMock()
    fake_registry = MagicMock()

    mock_request = Request(
        scope={
            "type": "http",
            "client": ("127.0.0.1", 12345),
            "headers": [],
            "path": "/ask/batch",
            "method": "POST",
            "query_string": b"",
        }
    )

    batch_req = BatchQuestionRequest(
        questions=[
            BatchQuestionItem(id="q1", question="ما هي الأهداف؟"),
            BatchQuestionItem(id="q2", question="من هو المدير؟"),
        ]
    )

    response = asyncio.run(
        routes.ask_questions_batch(
            request=mock_request,
            body=batch_req,
            qa_chain=fake_qa_chain,
            vector_store=fake_vector_store,
            registry=fake_registry,
        )
    )

    assert response.total_processed == 2
    assert len(response.answers) == 2
    assert response.answers[0].id == "q1"
    assert response.answers[0].response.confidence == 0.92
    assert fake_qa_chain.ask.call_count == 2
