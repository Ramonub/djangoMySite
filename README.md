## Django project setup guide

1. Update and upgrade Raspberry pi
```
sudo apt-get update && sudo apt-get upgrade
```

2. Go to directory where you want to install

3. Install git
```
sudo apt-get install git
```

4. Clone and enter repository
```
git clone https://github.com/Ramonub/djangoMySite.git && cd djangoMySite
```

5. Install virtualenv
```
pip install virtualenv
```

6. Create and activate venv
```
python -m virtualenv venv && source venv/bin/activate
```

7. Install mod_wsgi into python
```
cd mod_wsgi-4.9.4
python setup.py install && cd ..
```

8. Run setup file
```
python setup.py
```

9. Reboot to apply changes

10. Open django interface in webbrowser
```
http://192.168.252.32:8000/show
```
