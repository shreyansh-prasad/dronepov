import json
import logging

logger = logging.getLogger("Stage6_Manifest")

def generate_manifest(selected_frames, video_report, output_manifest_path, output_report_path):
    manifest_data = []
    for f in selected_frames:
        entry = {
            "frame_id": str(f['id']),
            "source_timestamp_ms": float(f['pts']),
            "usefulness_score": float(f.get('current_usefulness', 0.0)),
            "score_breakdown": {
                "marginal_coverage_gain": float(f.get('current_marginal_gain', 0.0)),
                "mission_value": float(f['mission_value']),
                "processing_cost": float(f['processing_cost'])
            },
            "low_confidence": bool(f.get('low_confidence', False)),
            "semantic_histogram": f.get('semantic_histogram', {}),
            "mask_path": str(f.get('mask_path', '')),
            "gate_reason": f.get('gate_reason', None)
        }
        manifest_data.append(entry)
        
    with open(output_manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
        
    with open(output_report_path, 'w') as f:
        json.dump(video_report, f, indent=2)
        
    logger.info(f"Manifest saved to {output_manifest_path}")
    logger.info(f"Report saved to {output_report_path}")
