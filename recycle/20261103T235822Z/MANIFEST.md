# Recycle manifest 20261103T235822Z

- `scripts/reproduce/d1_meshes_v6.sh`: gated repack for a D1 v6 that does not exist. G3's packed veto failed
  (`outputs/mesh_generative/g3/summary.md`), so the script could never run. Its function `repack` left `g3_pack.py`.
