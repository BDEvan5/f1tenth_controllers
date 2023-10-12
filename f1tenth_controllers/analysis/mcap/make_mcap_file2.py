
import json
import sys
from time import time_ns
from foxglove_schemas_protobuf import Pose_pb2

from mcap.writer import Writer
from mcap.well_known import SchemaEncoding, MessageEncoding
import numpy as np
import yaml, csv
from PIL import Image

import struct 
import base64

file_name = "Logs/test.mcap"
vehicle_name = "MapsMPCC"
map_name = "SaoPaulo"


class MapData:
    def __init__(self, map_name):
        self.path = "maps/"
        self.map_name = map_name

        self.xs, self.ys = None, None

        self.N = 0
        self.map_resolution = None
        self.map_origin = None
        self.map_img = None
        self.map_height = None
        self.map_width = None

        self.load_map_img()
        self.load_centerline()

    def load_map_img(self):
        with open(self.path + self.map_name + ".yaml", 'r') as file:
            map_yaml_data = yaml.safe_load(file)
            self.map_resolution = map_yaml_data["resolution"]
            self.map_origin = map_yaml_data["origin"]
            map_img_name = map_yaml_data["image"]

        self.map_img = np.array(Image.open(self.path + map_img_name).transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(np.float64)

        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 1.

        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]
        
    def load_centerline(self):
        track = np.loadtxt(self.path + self.map_name + "_centerline.csv", delimiter=',', skiprows=1)
        self.wpts = track[:, :2]
        self.N = len(self.wpts)
        


def load_agent_test_data(file_name):
    data = np.load(file_name)

    return data[:, :7], data[:, 7:]

states, actions = load_agent_test_data(f"Logs/{vehicle_name}/SimLog_{map_name}_maps_0.npy")
map_data = MapData(map_name)

with open(file_name, "wb") as stream:
    writer = Writer(stream)

    writer.start('x-jsonschema')

    with open(f"f1tenth_controllers/analysis/schemas/PoseInFrame.json", "rb") as f:
        schema = f.read()

    schema_id = writer.register_schema(
        name="foxglove.PoseInFrame",
        encoding=SchemaEncoding.JSONSchema,
        data=schema,
    )

    pose_channel_id = writer.register_channel(
        topic="vehicle_pose",
        message_encoding=MessageEncoding.JSON,
        schema_id=schema_id,
    )

    with open(f"f1tenth_controllers/analysis/schemas/Grid.json", "rb") as f:
        schema = f.read()

    schema_id = writer.register_schema(
        name="foxglove.Grid",
        encoding=SchemaEncoding.JSONSchema,
        data=schema,
    )

    grid_channel_id = writer.register_channel(
        topic="GridMap",
        message_encoding=MessageEncoding.JSON,
        schema_id=schema_id,
    )
    start_time = time_ns()

    grid = {}
    grid["timestamp"] = {
            "sec": int(start_time * 1e-9),
            "nsec": int(start_time - int(start_time * 1e-9)),
        }
    grid["pose"] = {
            "position": {"x": 0, "y": 0, "z": 0},
            "orientation": {"x": 0, "y": 0, "z": 0, "w": 0},
        }
    grid["frame_id"] = "o"
    grid["column_count"] = map_data.map_img.shape[0]
    grid["cell_size"] = {"x": 0.05, "y": 0.05}
    grid["row_stride"] = 4 * map_data.map_img.shape[1]
    grid["cell_stride"] = 4
    grid["fields"] = [
        {"name": "x", "offset": 0, "type": 7},
    ]
    grid["data"] = base64.b64encode(map_data.map_img).decode("utf-8")

    writer.add_message(
        channel_id=grid_channel_id,
        log_time=int(start_time),
        data=json.dumps(grid).encode("utf-8"),
        publish_time=int(start_time),
    )

    timestep = 0.05
    for i in range(600):
        time = start_time + i * 1e9 * timestep
        time_in_s = int(time * 1e-9)
        time_in_ns = int(time - time_in_s)

        pose = {}
        qz = np.sin(states[i, 4]/2)
        qw = np.cos(states[i, 4]/2)
 
        pose["pose"] = {
            "position": {"x": states[i, 0], "y": states[i, 1], "z": i},
            "orientation": {"x": 0, "y": 0, "z": qz, "w": qw},
        }
        pose["frame_id"] = "map"
        start_time = time_ns()
        pose["timestamp"] = {
            "sec": time_in_s,
            "nsec": time_in_ns,
        }
        writer.add_message(
            channel_id=pose_channel_id,
            log_time=int(time),
            data=json.dumps(pose).encode("utf-8"),
            publish_time=int(time),
        )

    writer.finish()

    # print(start_time)