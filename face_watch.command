#!/bin/bash
cd "$(dirname "$0")"
.venv/bin/python face_watch.py
echo; read -n1 -p "Press any key to close..."
