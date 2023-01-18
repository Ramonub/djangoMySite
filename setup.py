# import shutil
import os

print('===== Install requirements')
os.system('pip install -r requirements.txt')

print('===== Moving services to the right place')
os.system('sudo cp programFiles/django.service /usr/lib/systemd/system/')
os.system('sudo cp programFiles/django_worker.service /usr/lib/systemd/system/')

print('===== Activate services')
os.system('sudo systemctl enable /usr/lib/systemd/system/django.service')
os.system('sudo systemctl enable /usr/lib/systemd/system/django_worker.service')

if (os.path.exists('/usr/lib/systemd/system/django.service') and os.path.exists('/usr/lib/systemd/system/django_worker.service')):
    print('Files succesfully moved')
else:
    print('Something went wrong, run commands by hand: ')
    print('sudo cp programFiles/django.service /usr/lib/systemd/system/ && sudo cp programFiles/django_worker.service /usr/lib/systemd/system/')
    print('sudo systemctl enable /usr/lib/systemd/system/django.service && sudo systemctl enable /usr/lib/systemd/system/django_worker.service')
    