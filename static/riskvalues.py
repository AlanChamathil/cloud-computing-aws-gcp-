#!/usr/bin/python3
import cgitb
import cgi
import json
import random

cgitb.enable()

form = cgi.FieldStorage()
mean = float(form.getvalue("mean"))
std = float(form.getvalue("std"))
shots = int(form.getvalue("shots"))
t = form.getvalue("t")
simulated = [random.gauss(mean, std) for _ in range(shots)]

if t == 'sell':
    simulated.sort(reverse=False)
else:
    simulated.sort(reverse=True)
    
var95_value = simulated[int(len(simulated) * 0.95)]
var99_value = simulated[int(len(simulated) * 0.99)]

print("Content-Type: application/json;charset=utf-8")
print("")

# Create a dictionary with the message
response_data = {"var95": var95_value, "var99": var99_value}

# Output the JSON
print(json.dumps(response_data))