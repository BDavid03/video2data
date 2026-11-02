# Amendment 4: descriptive proxy failure analysis

Dated **2026-11-02**. Committed before any diagnostic value is computed. It extends Amendment 3 of
`docs/chain/mesh-selection-protocol.md` (the spoon silhouette proxy, verdict NOT VALIDATED). It lives in
its own file because the Amendment-3 request pins that file's hash. Drafted by the lead, then rewritten by
a Codex route call (`outputs/chain/codex/*-route-mechanism-10.answer.md`), which replaced the causal
labels with descriptive signatures.

Post-result diagnostic, fixed before new diagnostic values; Amendment 3 remains NOT VALIDATED and packed meshes remain unchanged.
Failed objects: sweeper, pitcher, lid; contrast: beige cup, blue cup, dust pan, dish rack, pot; exclude wood blocks and spoon.
Use only their Amendment-3 candidates, views, existing CD-O means and first-run ledger; no scans, new CD-O or episodes 41/43/44.
Pin baseline request, ledger, input hashes, renderer sources and package versions; missing or conflicting inputs stop analysis.
Replay selected yaw using ledger placement directly, original clipping and rasterization; no plane refit, placement search or yaw reselection.
Require exact parsed ledger IoU, selected projected area and target area; any mismatch stops analysis before pair summaries.
Let q_iv be recorded target IoU, s_i its equal-view mean, and c_i the recorded CD-O mean in cm.
For every unordered within-object pair compute d_v=1-IoU(P_iv,P_jv), its mean and maximum; two empty projections have d_v=0.
Near-projection pairs have max(d_v)<0.05; outline-changing pairs have max(d_v)>=0.05; neither label describes unseen views.
Eligible pairs have abs(c_i-c_j)>=0.03 cm; this is a practical gap, not a statistical significance claim.
A proxy tie has abs(s_i-s_j)<=1e-12; otherwise concordance requires (s_i-s_j)*(c_i-c_j)<0.
Per object/class, concordance is correct orders/all eligible pairs; report correct, reversed and tied counts separately.
Failures are reversed orders plus ties; H is near-projection failures/all failures; zero denominators are undefined.
Projection-indistinguishability signature requires >=3 failures, >=3 outline-changing eligible pairs, H>=2/3 and outline concordance>=0.70.
Outline-score disagreement requires >=3 outline-changing eligible pairs and outline concordance<0.50; otherwise label not separated.
Repeat eligibility and classification at 0.06 cm; differing classifications become not separated (gap-sensitive).
A failed-group signature requires all three failed objects to retain that signature; otherwise report not separated with per-object results.
Report equal-object means of defined fractions with contributing-object counts; pairs and views are not independent replicates.
For each object, B=mean_v(max_i q_iv); T=mean tau-b across all unordered view pairs using their common full candidate set.
T is undefined with fewer than two views or any undefined view-pair tau; group B/T differences require every object's value.
Flag failed-minus-passing mean B<=-0.10 or mean T<=-0.30 descriptively; neither flag changes classifications or identifies mask error.
For eligible outline-changing pairs report max_v(abs(q_iv-q_jv)) and abs(s_i-s_j), marking <=0.01 gaps as score compression.
Report original candidate extents and applied scales; retain continuous distances, gaps and signed per-view score differences.
Use a CPU-only resumable slice operation, separate run.json, replay ledger and pair tables under outputs/mesh_extents/proxy-insights/.
All signatures remain conditional on recorded views, normalization and selected yaw; hidden geometry versus mask/placement error remains unproven.
