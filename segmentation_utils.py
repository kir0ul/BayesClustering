#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import json
import cv2
import imageio.v3 as iio
from PIL import Image
import numpy as np
import h5py


def extract_eef_data_from_rosbag(bagfile, threshold=0.6):
    print(f"Extracting TF & gripper data from Bag file: `{bagfile}`")
    tf = {"x": [], "y": [], "z": [], "timestamp": []}
    gripper = {"val": [], "timestamp": []}

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS1_NOETIC)

    # Create reader instance and open for reading.
    with AnyReader([bagfile], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == "/imu_raw/Imu"]
        msg_nb_total = len(list(reader.messages(connections=connections)))
        for connection, timestamp, rawdata in tqdm(
            reader.messages(connections=connections), total=msg_nb_total
        ):
            msg = reader.deserialize(rawdata, connection.msgtype)
            # print(msg.header.frame_id
            if connection.msgtype == "tf2_msgs/msg/TFMessage":
                if (
                    msg.transforms[0].child_frame_id == "tool0_controller"
                    and msg.transforms[0].header.frame_id == "base"
                ):
                    tf["x"].append(msg.transforms[0].transform.translation.x)
                    tf["y"].append(msg.transforms[0].transform.translation.y)
                    tf["z"].append(msg.transforms[0].transform.translation.z)
                    # tf["timestamp"].append(pd.to_datetime(timestamp, utc=True).tz_localize("UTC").tz_convert("EST"))
                    tf["timestamp"].append(
                        pd.to_datetime(timestamp, utc=True).tz_convert("EST")
                    )

            if connection.msgtype == "ur5e_move/msg/gripper_pos":
                gripper["val"].append(msg.gripper_pos)
                gripper["timestamp"].append(
                    pd.to_datetime(timestamp, utc=True).tz_convert("EST")
                )

    tf_df = pd.DataFrame(tf)
    gripper_df = pd.DataFrame(gripper)
    gripper_df["val"] = gripper_df["val"].apply(lambda elem: elem / 100)

    # Merge both DataFrames into one
    traj = pd.merge_asof(tf_df, gripper_df, on="timestamp")
    traj.dropna(inplace=True, ignore_index=True)
    traj.rename(columns={"val": "gripper"}, inplace=True)

    # # Simplify gripper data
    # traj.loc[traj.gripper < threshold, "gripper"] = 0
    # traj.loc[traj.gripper >= threshold, "gripper"] = 1

    print("Extracting TF & gripper data from Bag file: done ✓")
    return traj


def json2dict(ground_truth_segm_file):
    if not ground_truth_segm_file.exists():
        print(
            "JSON ground truth segmentation file not found:\n"
            f"`{ground_truth_segm_file}`"
        )
        return

    # Load JSON as dict
    with open(ground_truth_segm_file) as fid:
        json_str = fid.read()
    json_dict = json.loads(json_str)

    return json_dict


def get_bagfiles_from_json(ground_truth_segm_file):
    json_dict = json2dict(ground_truth_segm_file)
    root_path = Path(json_dict.get("root_path"))
    bagfiles = []
    for item in json_dict.get("groundtruth"):
        bagpath = root_path / item.get("filename")
        bagfiles.append(bagpath)
    return sorted(bagfiles)


def get_ground_truth_segmentation(ground_truth_segm_file, bagfile):
    json_dict = json2dict(ground_truth_segm_file)

    # Find segmention ground truth from file
    gt_segm_dict = None
    for item in json_dict.get("groundtruth"):
        if item.get("filename") == bagfile.name:
            gt_segm_dict = item
            break

    if gt_segm_dict is None:
        print(f"Segmentation data not found in `{ground_truth_segm_file}`")

    return gt_segm_dict


def extract_video_from_bag(bagfile, fps=20):
    print("Extracting video from Bag file...")

    extension = "mkv"
    video_path = bagfile.parent / (bagfile.stem + "." + extension)

    # Initialize video writer
    img_height, img_width, _ = get_img_height_width(bagfile=bagfile)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'mp4v' for MP4 format
    video = cv2.VideoWriter(
        filename=video_path, fourcc=fourcc, fps=fps, frameSize=(img_width, img_height)
    )

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS1_NOETIC)

    # Create reader instance and open for reading.
    with AnyReader([bagfile], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == "/imu_raw/Imu"]
        msg_nb_total = len(list(reader.messages(connections=connections)))
        for connection, timestamp, rawdata in tqdm(
            reader.messages(connections=connections), total=msg_nb_total
        ):
            msg = reader.deserialize(rawdata, connection.msgtype)
            # print(msg.header.frame_id
            if connection.msgtype == "sensor_msgs/msg/Image":
                frame = msg.data.reshape((msg.height, msg.width, 3))

                current_ts = pd.to_datetime(timestamp, utc=True).tz_convert("EST")

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Add timestamp overlay on image
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt1 = current_ts.isoformat()
                txt2 = str(current_ts.timestamp())
                fontScale = 0.5
                white = (255, 255, 255)
                fontthickness = 1
                cv2.putText(
                    img=img,
                    text=txt1,
                    org=(10, 30),
                    fontFace=font,
                    fontScale=fontScale,
                    color=white,
                    thickness=fontthickness,
                )
                cv2.putText(
                    img=img,
                    text=txt2,
                    org=(10, 50),
                    fontFace=font,
                    fontScale=fontScale,
                    color=white,
                    thickness=fontthickness,
                )

                # Add images to the video
                video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Release the video writer
    video.release()
    cv2.destroyAllWindows()
    print("Extracting video from Bag file: done ✓")
    print(f"Video path: {video_path}")
    return video_path


def get_video_frame(index, video_path):
    # read a single frame
    try:
        frame = iio.imread(
            video_path,
            index=index,
            plugin="pyav",
        )
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # return img
        return frame
    except StopIteration:
        print("Reached the end of the video file")
        return np.asarray(Image.new("RGB", (3840, 2160), (0, 0, 0)))
        # return np.asarray(Image.new("RGB", (720, 405), (0, 0, 0)))


def get_img_height_width(bagfile):
    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS1_NOETIC)

    # Create reader instance and open for reading.
    with AnyReader([bagfile], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == "/imu_raw/Imu"]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            # print(msg.header.frame_id
            if connection.msgtype == "sensor_msgs/msg/Image":
                # print(msg)
                break
    return msg.height, msg.width, msg.data


def get_gtdict_filenames(ground_truth_segm_file, filenum):
    if not ground_truth_segm_file.exists():
        print(
            "JSON ground truth segmentation file not found:\n"
            f"`{ground_truth_segm_file}`"
        )
        return

    # Load JSON as dict
    with open(ground_truth_segm_file) as fid:
        json_str = fid.read()
    json_str = json.loads(json_str)
    gt_array = json_str.get("groundtruth")
    gt_segm_dict = gt_array[filenum]
    video_file = gt_segm_dict.get("video")
    hdf5_file = gt_segm_dict.get("hdf5")
    return gt_segm_dict, video_file, hdf5_file


def read_h5_data(fname):
    hf = h5py.File(fname, "r")
    print(list(hf.keys()))
    js = hf.get("joint_state_info")
    joint_time = np.array(js.get("joint_time"))
    joint_pos = np.array(js.get("joint_positions"))
    joint_vel = np.array(js.get("joint_velocities"))
    joint_eff = np.array(js.get("joint_effort"))
    joint_data = [joint_time, joint_pos, joint_vel, joint_eff]

    tf = hf.get("transform_info")
    tf_time = np.array(tf.get("transform_time"))
    tf_pos = np.array(tf.get("transform_positions"))
    tf_rot = np.array(tf.get("transform_orientations"))
    tf_data = [tf_time, tf_pos, tf_rot]
    # print(tf_pos)

    wr = hf.get("wrench_info")
    wrench_time = np.array(wr.get("wrench_time"))
    wrench_frc = np.array(wr.get("wrench_force"))
    wrench_trq = np.array(wr.get("wrench_torque"))
    wrench_data = [wrench_time, wrench_frc, wrench_trq]

    gp = hf.get("gripper_info")
    gripper_time = np.array(gp.get("gripper_time"))
    gripper_pos = np.array(gp.get("gripper_position"))
    # gripper_time = []
    # gripper_pos = []
    gripper_data = [gripper_time, gripper_pos]

    hf.close()

    return joint_data, tf_data, wrench_data, gripper_data


def ts2df(tf_data, gripper_data):
    # ts_time = tf_data[0][:, 0] + tf_data[0][:, 1] * (10.0**-9)

    def conv2timestamps(tarray):
        # nanosec_conv = 1e-9
        time_sec = tarray[:, 0]
        time_nanosec = tarray[:, 1]
        # time = time_sec + time_nanosec * nanosec_conv

        timestamps = []
        for t_idx, t_val in enumerate(time_sec):
            timestamp = pd.Timestamp(
                time_sec[t_idx], unit="s", tz="EST"
            ) + pd.to_timedelta(time_nanosec[t_idx], unit="ns")
            timestamps.append(timestamp)
        timestamps = pd.Series(timestamps)
        return timestamps

    tf_df = pd.DataFrame(
        {
            "pos_x": tf_data[1][:, 0],
            "pos_y": tf_data[1][:, 1],
            "pos_z": tf_data[1][:, 2],
            "timestamp": conv2timestamps(tf_data[0]),
        }
    )

    gripper_df = pd.DataFrame(
        {
            "val": gripper_data[1].squeeze(),
            "timestamp": conv2timestamps(gripper_data[0]),
        }
    )
    gripper_df["val"] = gripper_df["val"].apply(lambda elem: elem / 100)

    # Merge both DataFrames into one
    traj = pd.merge_asof(tf_df, gripper_df, on="timestamp")
    traj.dropna(inplace=True, ignore_index=True)
    traj.rename(columns={"val": "gripper"}, inplace=True)
    return traj
