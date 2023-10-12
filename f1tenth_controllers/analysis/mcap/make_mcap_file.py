
import json
import sys
from time import time_ns

from mcap.writer import Writer
from mcap.well_known import SchemaEncoding, MessageEncoding, TopicType
import numpy as np


file_name = "Logs/test.mcap"
vehicle_name = "MapsMPCC"
map_name = "SaoPaulo"


def load_agent_test_data(file_name):
    data = np.load(file_name)

    return data[:, :7], data[:, 7:]

states, actions = load_agent_test_data(f"Logs/{vehicle_name}/SimLog_{map_name}_maps_0.npy")

with open(file_name, "wb") as stream:
    writer = Writer(stream)

    writer.start()

    schema_id = writer.register_schema(
        name="pose",
        encoding="jsonschema",
        data=json.dumps(
            {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                    },
                    "y": {
                        "type": "number",
                    }
                },
            }
        ).encode(),
    )

    pose_channel_id = writer.register_channel(
        schema_id=schema_id,
        topic="pose",
        message_encoding="json",
    )

    schema_id = writer.register_schema(
        name="steering",
        encoding="jsonschema",
        data=json.dumps(
            {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": "number",
                    }
                },
            }
        ).encode(),
    )
    steering_channel_id = writer.register_channel(
        schema_id=schema_id,
        topic="steering",
        message_encoding="json",
    )

    start_time = time_ns()
    timestep = 0.05
    for i in range(500):
        time = start_time + i * 1e9 * timestep
        writer.add_message(
            channel_id=pose_channel_id,
            log_time=int(time),
            data=json.dumps({"x": states[i, 0], "y": states[i, 1]}).encode("utf-8"),
            publish_time=int(time),
        )
        steering_angle = np.rad2deg(actions[i, 0])
        writer.add_message(
            channel_id=steering_channel_id,
            log_time=int(time),
            data=json.dumps({"angle": steering_angle}).encode("utf-8"),
            publish_time=int(time),
        )

    writer.finish()

    # print(start_time)