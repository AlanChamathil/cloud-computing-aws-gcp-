#!/bin/bash
yum update -y
yum install httpd -y
service httpd start
chkconfig httpd on

wget https://clouding-computing-414213.nw.r.appspot.com/cacheavoid/index.html -P /var/www/html
wget https://clouding-computing-414213.nw.r.appspot.com/cacheavoid/riskvalues.py -P /var/www/cgi-bin
chmod +x /var/www/cgi-bin/riskvalues.py