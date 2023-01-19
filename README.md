## Django project setup guide on new Raspberry Pi

1. Update and upgrade Raspberry pi
```
sudo apt-get update && sudo apt-get upgrade
```

2. Go to directory where you want to install

3. Install git, apache2, redis-server, daphne and python requirements
```
cd djangoMySite

sudo apt-get install git
sudo apt install apache2
sudo apt install redis-server
sudo apt-get install daphne

pip install -r requirements.txt
```

4. Redis Setup
```
sudo systemctl enable redis
```
Change 'bind' in /etc/redis/redis.conf to the ip adres of the hosting device:  192.168.252.32

5. Apache2 setup
Change Root to serve static files
``` 
sudo nano /etc/apache2/sites-available/000-default.conf
```
Change DocumentRoot to path to static files: /path/to/static/files\
``` 
sudo nano /etc/apache2/apache2.conf
```
Find: <Directory {dir}> ... </Directory ...>\
Change dir to static files directory: /path/to/static/files/\
Restart Apache
```
sudo systemctl restart apache2.service
```

6. MCCDAQ library setup
```
cd ~/
git clone https://github.com/mccdaq/daqhats.git
cd ~/daqhats
sudo ./install.sh
cp -r /home/$USER/daqhats/daqhats /home/$USER/djangoMySite/mysite
```

7. Clone and enter repository
```
git clone https://github.com/Ramonub/djangoMySite.git && cd djangoMySite
```

8. Create and activate venv
```
python -m virtualenv venv && source venv/bin/activate
```

9. Install mod_wsgi into python
```
cd mod_wsgi-4.9.4
python setup.py install && cd ..
```

10. Run setup file.\
If some packages wont download try to download them manually: pip install {package name}.\
Some packages can take a long time to download, so be patient.
```
python setup.py
```

11. Reboot to apply changes

12. Open django interface in webbrowser
```
http://192.168.252.32:8000/show
```


NOTE:
Everything is configured with user madern\
If the user changed some files need to be changed as well:\

Systemd files:\
- /usr/lib/systemd/system/django.service\
- /usr/lib/systemd/system/django_worker.service\

Apache2 files:\
- /etc/apache2/sites-available/000-default.conf\
- /etc/apache2/apache2.conf\