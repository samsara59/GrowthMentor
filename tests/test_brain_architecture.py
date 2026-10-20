import unittest

from app.brain.hierarchy import validate_hierarchy
from app.brain.messages import AgentMessage
from app.brain.message_bus import AgentBus


class TestBrainArchitecture(unittest.TestCase):

    def test_hierarchy(self):
        self.assertTrue(validate_hierarchy())

    def test_agent_handoff(self):
        bus = AgentBus()

        def evidence_agent(request):
            return AgentMessage(
                session_id=request.session_id,
                request_id=request.request_id,
                sender="evidence",
                recipient="supervisor",
                message_type="review_result",
                parent_id=request.message_id,
                payload={"verdict": "uncertain"},
            )

        bus.register("evidence", evidence_agent)

        request = AgentMessage(
            session_id="session1",
            request_id="turn1",
            sender="supervisor",
            recipient="evidence",
            message_type="task_request",
            payload={"task": "verify"},
        )

        response = bus.dispatch(request)

        self.assertEqual(
            response.payload["verdict"], "uncertain"
        )
        self.assertEqual(
            len(bus.trace("session1", "turn1")), 2
        )

    def test_unknown_agent(self):
        bus = AgentBus()
        request = AgentMessage(
            session_id="s",
            request_id="r",
            sender="supervisor",
            recipient="unknown",
            message_type="task_request",
        )

        with self.assertRaises(KeyError):
            bus.dispatch(request)

    def test_reject_wrong_request(self):
        bus = AgentBus()

        def incorrect(request):
            return AgentMessage(
                session_id="other",
                request_id=request.request_id,
                sender="worker",
                recipient="supervisor",
                message_type="review_result",
                parent_id=request.message_id,
            )

        bus.register("worker", incorrect)

        request = AgentMessage(
            session_id="s",
            request_id="r",
            sender="supervisor",
            recipient="worker",
            message_type="task_request",
        )

        with self.assertRaises(ValueError):
            bus.dispatch(request)


if __name__ == "__main__":
    unittest.main()
