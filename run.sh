#!/bin/sh
# Запуск консоли команды (console/server.py). Аргументы пробрасываются в него же:
# --host, --port.
#
# Порт проверяем до старта: uvicorn занятый порт не отбирает, а пишет
# «address already in use» и оставляет жить старый процесс — правки при этом
# молча уезжают в никуда (см. console/README.md, «Грабли»).
set -eu

cd "$(dirname "$0")/console"

port=8766
prev=""
for arg in "$@"; do
	if [ "$prev" = "--port" ]; then
		port=$arg
	fi
	prev=$arg
done

pid=$(lsof -ti "tcp:$port" -sTCP:LISTEN || true)
if [ -n "$pid" ]; then
	echo "Порт $port уже занят:"
	ps -o pid,lstart,command -p $pid | tail -n +2
	echo
	echo "Останови старый процесс и повтори:  kill $pid"
	exit 1
fi

# Браузер открываем из фона, дождавшись первого ответа сервера: сразу после
# запуска порт ещё не слушается и вкладка встретила бы «не удалось подключиться».
# Ждём не дольше 30 с — если не поднялся, молча не открываем.
url="http://127.0.0.1:$port/"
if [ -n "${CREW_CONSOLE_TOKEN:-}" ]; then
	# С токеном страница без ?t= отдаёт 401 — открываем сразу со ссылкой-пропуском.
	url="$url?t=$CREW_CONSOLE_TOKEN"
fi
(
	n=0
	while [ "$n" -lt 60 ]; do
		if curl -s -m 1 -o /dev/null "http://127.0.0.1:$port/"; then
			open "$url"
			exit 0
		fi
		n=$((n + 1))
		sleep 0.5
	done
) &

exec uv run server.py "$@"
