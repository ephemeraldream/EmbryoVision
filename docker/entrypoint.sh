#!/bin/sh
cd /code/backend/depdl/

echo '1. Collecting static files ...'
python manage.py collectstatic --noinput --clear

echo '2. Synchronizing migrations ...'
python manage.py makemigrations
python manage.py migrate

echo '3. Starting Django with gunicorn...'
# tail -f /dev/null
daphne -b 0.0.0.0 -p 8000 depdl.asgi:application

