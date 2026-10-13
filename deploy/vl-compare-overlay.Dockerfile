ARG BASE_IMAGE=yoloe_service_test:human-review-20260907-v1
FROM ${BASE_IMAGE}
COPY vl_compare.py /app/api/vl_compare.py
COPY history_compare_runtime /app/history_compare_runtime
