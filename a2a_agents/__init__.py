"""The agents as network services.

    monitoring   A2A agent, skill ran.monitoring.observe          (:8300)
    near_rt_ric  A2A agent, skill ran.nearrt.mitigate             (:8301)
    non_rt_ric   A2A agent, skill ran.nonrt.propose_policy        (:8302)
    governance   HTTP policy decision point, POST /authorize      (:8310)

Agents talk to each other only over A2A (a2a-sdk, JSON-RPC binding). The
governance service is not an A2A peer: it is the policy decision point that
the RIC agents' tool middleware (the enforcement point) must consult before
every protected call, so it is a plain request/response service.

    python -m a2a_agents governance  --port 8310
    python -m a2a_agents near_rt_ric --port 8301
    python -m a2a_agents non_rt_ric  --port 8302
    python -m a2a_agents monitoring  --port 8300
"""
