"""BEHAVE paper metrics, preserving the recorded coordinate and averaging rules."""
import hashlib
import numpy as np
from scipy.spatial import cKDTree

from .metric_kernels import rigid_align, chamfer_distance


def contact_counts(human, obj, labels):
    labels = np.asarray(labels)
    if labels.shape != (len(human),):
        raise ValueError("Contact labels must match human vertices")
    predicted = cKDTree(obj).query(human)[0] < .05
    valid = (labels == 0) | (labels == 1)
    positive = labels == 1
    return dict(tp=int((predicted & positive & valid).sum()),
                fp=int((predicted & ~positive & valid).sum()),
                fn=int((~predicted & positive & valid).sum()))


def surface_cd(human, gt_human, obj, gt_obj, human_faces, object_faces, ann_id, seed=123):
    import trimesh
    state = np.random.get_state()
    try:
        sample_seed = int.from_bytes(hashlib.sha256(f"{seed}:{ann_id}:formal".encode()).digest()[:4], "little")
        np.random.seed(sample_seed)
        aligned = rigid_align(np.concatenate((human, obj)), np.concatenate((gt_human, gt_obj)))
        ph = aligned[:len(human)].astype(np.asarray(human).dtype, copy=False)
        po = aligned[len(human):].astype(np.asarray(obj).dtype, copy=False)
        meshes = [trimesh.Trimesh(v, f, process=False, maintain_order=True) for v, f in (
            (ph, human_faces), (gt_human, human_faces), (po, object_faces), (gt_obj, object_faces))]
        samples = [mesh.sample(6000) for mesh in meshes]
        return dict(cd_human_cm=float(chamfer_distance(samples[1], samples[0])) * 100,
                    cd_object_cm=float(chamfer_distance(samples[3], samples[2])) * 100)
    finally:
        np.random.set_state(state)


def paper_summary(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    precisions = [r["tp"] / (r["tp"] + r["fp"]) for r in rows if r["tp"] + r["fp"]]
    recalls = [r["tp"] / (r["tp"] + r["fn"]) for r in rows if r["tp"] + r["fn"]]
    p = float(np.mean(precisions)) if precisions else 0.
    r = float(np.mean(recalls)) if recalls else 0.
    return dict(sample_count=len(rows), cd_human_cm=float(np.mean([x["cd_human_cm"] for x in rows])),
                cd_object_cm=float(np.mean([x["cd_object_cm"] for x in rows])),
                precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.,
                precision_valid_frames=len(precisions), recall_valid_frames=len(recalls))


def evaluate_batch(output, targets, metadata, seed=123):
    from .reconstruction.models.templates import smplh, obj_dict
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    rows = []
    for i, ann_id in enumerate(array(metadata["ann_id"]).tolist()):
        raw_human = array(output["smpl_verts6000"])[i]
        # Match the published inherited evaluator: human pelvis centering with
        # the raw object R/t, before joint similarity alignment for CD only.
        human = raw_human - np.asarray(smplh.joint_regressor[smplh.root_joint_idx]) @ raw_human
        mesh = obj_dict[metadata["obj_name"][i]].load_template()
        obj = mesh.vertices @ array(output["obj_pose"])[i].T + array(output["obj_trans"])[i]
        gt_obj = mesh.vertices @ array(targets["obj_pose"])[i].T + array(targets["obj_trans"])[i]
        human_key = "smpl_mesh_cam" if "smpl_mesh_cam" in targets else "smpl_verts"
        gt_human = array(targets[human_key])[i]
        labels = array(targets["h_contacts"])[i]
        if labels.shape != (6890,) or not all(np.isfinite(a).all() for a in (human, obj, gt_human, gt_obj)):
            raise ValueError("Expected finite complete BEHAVE test meshes and 6890 contact labels")
        rows.append(dict(ann_id=int(ann_id), object_name=metadata["obj_name"][i],
            **surface_cd(human, gt_human, obj, gt_obj, smplh.faces, mesh.faces, ann_id, seed),
            **contact_counts(human, obj, labels)))
    return rows

