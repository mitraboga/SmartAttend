from src.database import list_recent_attendance, upsert_attendance


class AttendanceLogger:
    """Compatibility wrapper around the SQLite attendance store."""

    def mark_attendance(
        self,
        *,
        student_id: int,
        status: str,
        confidence: float,
        liveness_score: float,
        note: str,
        raw_label: str,
        claimed_roll_no: str,
    ) -> tuple[bool, str]:
        action, _record = upsert_attendance(
            student_id=student_id,
            status=status,
            confidence=confidence,
            liveness_score=liveness_score,
            note=note,
            raw_label=raw_label,
            claimed_roll_no=claimed_roll_no,
        )
        return action != "duplicate", action

    def recent_records(self, limit: int = 20) -> list[dict]:
        return list_recent_attendance(limit=limit)
