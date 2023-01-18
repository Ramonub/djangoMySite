## Django project setup guide

1. Update and upgrade Raspberry pi
-> sudo apt-get update && sudo apt-get upgrade

2. Go to directory where you want to install

3. Install git
-> sudo apt-get install git

4. Clone and enter repository
-> git clone https://github.com/Ramonub/djangoMySite.git
-> cd djangoMySite

5. Install virtualenv
-> pip install virtualenv

6. Create and activate venv
-> virtualenv venv -p python && source venv/bin/activate

7. Run setup file
-> python setup.py

8. Open django interface in webbrowser
-> http://192.168.252.32:8000/show
