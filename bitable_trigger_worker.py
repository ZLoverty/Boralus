"""
多维表格「按钮字段触发本地脚本」方案(无需公网)

原理(飞书按钮无法直接调本地服务,需借自动化中转):
  真按钮字段(点击)
    → 自动化流程(触发条件「点击按钮时」)
    → 流程动作:勾选当前记录的「已请求」复选框 ← 自动化写记录是允许的
    → 本地脚本轮询发现「已请求」= true
    → 提取 品名/喷嘴温度/最大体积流速 → 存成本地 JSON 文件
    → 把「状态」更新为「审核中」→ 复位「已请求」

表格里需要的列:
  - 「提交审核」按钮字段:绑定自动化流程(触发「点击按钮时」),
    流程加「更新记录」动作:把「已请求」字段设为勾选
  - 「已请求」复选框:被自动化勾选、被脚本复位(可放到视图隐藏列)
  - 「品名」文本 / 「喷嘴温度」数字 / 「最大体积流速」数字
  - 「状态」文本

运行:  .venv/bin/python bitable_trigger_worker.py
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import lark_oapi as lark
from dotenv import load_dotenv
from lark_oapi.api.bitable.v1 import (
    AppTableRecord,
    ListAppTableRecordRequest,
    UpdateAppTableRecordRequest,
)

load_dotenv(Path(__file__).parent / ".env")

APP_ID = os.environ["FEISHU_APP_ID"]
APP_SECRET = os.environ["FEISHU_APP_SECRET"]
APP_TOKEN = os.environ["FEISHU_APP_TOKEN"]
TABLE_ID = os.environ["FEISHU_TABLE_ID"]

SIGNAL_FIELD = "已请求"  # 复选框:自动化勾选(信号),脚本执行后复位
STATUS_FIELD = "状态"  # 文本列:审核状态
STATUS_REVIEWING = "审核中"  # 状态列写入的值
REQUIRED_FIELDS = ["品名", "喷嘴温度", "最大体积流速"]  # 需要写入 JSON 的字段
OUTPUT_DIR = Path(__file__).parent / "submitted"  # JSON 保存目录
POLL_INTERVAL = 5  # 轮询间隔(秒)

client = lark.Client.builder().app_id(APP_ID).app_secret(APP_SECRET).build()


def run_logic(record_id: str, fields: dict) -> tuple[bool, str]:
    """提取品名/喷嘴温度/最大体积流速 → 存本地 JSON。

    返回 (是否成功, 摘要文本);失败时不改「状态」。
    """
    missing = [k for k in REQUIRED_FIELDS if k not in fields or fields[k] is None]
    if missing:
        return False, f"缺少字段: {', '.join(missing)},请补全后重新点击"

    payload = {
        "品名": fields["品名"],
        "喷嘴温度": fields["喷嘴温度"],
        "最大体积流速": fields["最大体积流速"],
        "record_id": record_id,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 文件名含时间戳,重复提交同名品名也不会互相覆盖
    fname = f"{datetime.now():%Y%m%d_%H%M%S}_{record_id}.json"
    path = OUTPUT_DIR / fname
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return True, f"已保存: {path}"


def fetch_all_records():
    """分页拉取全部记录,返回 [(record_id, fields), ...]"""
    records = []
    page_token = None
    while True:
        builder = (
            ListAppTableRecordRequest.builder()
            .app_token(APP_TOKEN)
            .table_id(TABLE_ID)
            .page_size(500)
        )
        if page_token:
            builder.page_token(page_token)
        resp = client.bitable.v1.app_table_record.list(builder.build())
        if not resp.success():
            print(f"[拉取失败] code={resp.code} msg={resp.msg}")
            return records
        records += [(it.record_id, it.fields) for it in (resp.data.items or [])]
        if not resp.data.has_more:
            break
        page_token = resp.data.page_token
    return records


def update_record(record_id: str, fields: dict):
    """按给定字段更新一条记录"""
    body = AppTableRecord.builder().fields(fields).build()
    req = (
        UpdateAppTableRecordRequest.builder()
        .app_token(APP_TOKEN)
        .table_id(TABLE_ID)
        .record_id(record_id)
        .request_body(body)
        .build()
    )
    resp = client.bitable.v1.app_table_record.update(req)
    if not resp.success():
        print(f"[更新失败] record={record_id} code={resp.code} msg={resp.msg}")
    return resp.success()


def main():
    print(
        f"开始监听:按钮点击后「{SIGNAL_FIELD}」被自动化勾选即触发"
        f"(每 {POLL_INTERVAL}s 轮询,Ctrl+C 退出)"
    )
    while True:
        for record_id, fields in fetch_all_records():
            if fields.get(SIGNAL_FIELD) is True:
                print(f"[触发] record={record_id}")
                ok, summary = run_logic(record_id, fields)
                if ok:
                    update_record(
                        record_id, {STATUS_FIELD: STATUS_REVIEWING, SIGNAL_FIELD: False}
                    )
                    print(f"[完成] {summary} | 状态 → {STATUS_REVIEWING}")
                else:
                    # 数据不完整:不动状态,只复位信号,补全后可再次点击
                    update_record(record_id, {SIGNAL_FIELD: False})
                    print(f"[跳过] {summary}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已停止")
