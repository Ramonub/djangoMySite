import shutil
import os

os.system('sudo apt-get update && sudo apt-get upgrade')

os.system('pip install virtualenv && virtualenv venv -p python')

os.system('source venv/bin/activate')

os.system('pip install -r requirements.txt')


# Moving systemd to correct place
shutil.copy('programFiles/django.service', '/usr/lib/systemd/system/')
shutil.copy('programFiles/django_worker.service', '/usr/lib/systemd/system/')
# Enable services
os.system('sudo systemctl enable /usr/lib/systemd/system/django.service')
os.system('sudo systemctl enable /usr/lib/systemd/system/django_worker.service')