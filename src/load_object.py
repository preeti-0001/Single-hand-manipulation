import genesis as gs
from datetime import datetime


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


print(f"[{now()}] Starting Genesis initialization...")

gs.init(backend=gs.gpu)

print(f"[{now()}] Creating scene...")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -10.0)),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
    show_viewer=True,
)

print(f"[{now()}] Adding ground...")

scene.add_entity(gs.morphs.Plane())

print(f"[{now()}] Adding apple...")

apple = scene.add_entity(
    gs.morphs.Mesh(
        file="hrdexdb/assets/mesh/apple/apple.obj",
        pos=(0, 0, 0.1),
        fixed=False,
    )
)

print(f"[{now()}] Apple added. Starting scene.build()...")

build_start = datetime.now()

scene.build()

build_end = datetime.now()

print(f"[{now()}] scene.build() finished.")
print(f"Build time: {build_end - build_start}")

print(f"[{now()}] Starting simulation...")

sim_start = datetime.now()

for i in range(50):
    scene.step()

sim_end = datetime.now()

print(f"[{now()}] Simulation finished.")
print(f"Simulation time: {sim_end - sim_start}")

while True:
    scene.step()
