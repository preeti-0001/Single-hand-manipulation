import genesis as gs
import numpy as np
from scipy.spatial.transform import Rotation

object_name = "apple"


def load_object(object_name):
    OBJECT_MESH = f"hrdexdb/assets/mesh/{object_name}/{object_name}.obj"
    POSE_FILE = f"hrdexdb/allegro_v5/{object_name}/4/object_6d_pose.npz"

    #### Add check to see if the files exist


    # ============================================================
    # Load HRDexDB object trajectory
    # ============================================================

    data = np.load(POSE_FILE)

    print("Keys:", data.files)
    print("Number of frames:", len(data.files))

    # Convert frame_0, frame_1, ... into ordered list
    frame_keys = sorted(data.files, key=lambda x: int(x.split("_")[1]))

    poses = [data[key] for key in frame_keys]

    print("First pose:")
    print(poses[0])

    print("Last pose:")
    print(poses[-1])

    # ============================================================
    # Genesis
    # ============================================================


    scene = gs.Scene(
        show_viewer=True,
    )

    # ============================================================
    # Ground
    # ============================================================

    plane = scene.add_entity(gs.morphs.Plane())

    # ============================================================
    # Object
    # ===========================================================

    obj = scene.add_entity(
        gs.morphs.Mesh(
            file=OBJECT_MESH,
            pos=(0.0, 0.0, 0.1),
            scale=1.0,
            fixed=True,
        )
    )

    # ============================================================
    # Build scene
    # ============================================================

    scene.build()

    # ============================================================
    # Replay trajectory
    # ============================================================

    for i, T in enumerate(poses):

        # ============================================================
        # Paths
        # ============================================================

        # --------------------------------------------------------
        # 4x4 homogeneous transformation
        #
        # T =
        # [ R R R tx ]
        # [ R R R ty ]
        # [ R R R tz ]
        # [ 0 0 0  1 ]
        # --------------------------------------------------------

        position = T[:3, 3]

        rotation_matrix = T[:3, :3]

        # scipy gives quaternion as:
        # [x, y, z, w]
        quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()

        # Genesis uses:
        # [w, x, y, z]
        quat_wxyz = np.array(
            [
                quat_xyzw[3],
                quat_xyzw[0],
                quat_xyzw[1],
                quat_xyzw[2],
            ]
        )

        # --------------------------------------------------------
        # Move Apple
        # --------------------------------------------------------

        obj.set_pos(position)
        obj.set_quat(quat_wxyz)

        # Advance Genesis
        scene.step()

        print(f"Frame {i:03d} | " f"pos = {position} | " f"quat = {quat_wxyz}")


def main():
    object_name = input("Enter object name: ")
    
    gs.init(backend=gs.gpu)

    while object_name != "quit":
        load_object(object_name)
        object_name = input("Enter object name: ")

main();