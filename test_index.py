import time

from orca_sim import JointPoseWrapper, OrcaHandRight

env = JointPoseWrapper(OrcaHandRight(render_mode="human"))
obs, info = env.reset()

# viewer 띄우면서 미지정 관절은 0 rad
for _ in range(60):
    env.step({})
    time.sleep(1 / 30)

# 검지만 접기
for _ in range(120):
    env.step({"index_mcp": 1.0, "index_pip": 1.2})
    time.sleep(1 / 30)

env.close()
