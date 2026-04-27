"""Inject rename + delete report routes into app.py"""

NEW_ROUTES = r"""
# --- Report Management Routes -------------------------------------------------
@app.route("/api/reports/<report_id>/rename", methods=["PATCH"])
@login_required
def api_report_rename(report_id):
    if not MONGO_AVAILABLE:
        return jsonify({"error": "MongoDB unavailable"}), 503
    try:
        data = request.get_json()
        new_name = (data or {}).get("name", "").strip()
        if not new_name:
            return jsonify({"error": "Name cannot be empty"}), 400
        result = db["reports"].update_one(
            {"_id": ObjectId(report_id), "user_id": session["user_id"]},
            {"$set": {"filename": new_name}}
        )
        if result.matched_count == 0:
            return jsonify({"error": "Report not found"}), 404
        return jsonify({"ok": True, "filename": new_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reports/<report_id>", methods=["DELETE"])
@login_required
def api_report_delete(report_id):
    if not MONGO_AVAILABLE:
        return jsonify({"error": "MongoDB unavailable"}), 503
    try:
        result = db["reports"].delete_one(
            {"_id": ObjectId(report_id), "user_id": session["user_id"]}
        )
        if result.deleted_count == 0:
            return jsonify({"error": "Report not found"}), 404
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

"""

content = open("app.py", encoding="utf-8").read()

MARKER = 'if __name__ == "__main__":'
if MARKER not in content:
    MARKER = "if __name__ == '__main__':"

if "/api/reports/<report_id>/rename" in content:
    print("Routes already present.")
else:
    content = content.replace(MARKER, NEW_ROUTES + MARKER, 1)
    open("app.py", "w", encoding="utf-8").write(content)
    print("rename route added:", "/rename" in content)
    print("delete route added:", 'methods=["DELETE"]' in content)
