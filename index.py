import os,sys
import logging

from flask import Flask, jsonify, request, render_template, Response
import boto3, http.client, json, math, random
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from pandas_datareader import data as pdr
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, time
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

#Global Variables
scale_serv = {'s': None, 'r': 0}
lambda_warmup_results = []
h = 0
d = 0
t = ""
p = 0
profit_loss = []
var95 = []
var99 = []
lambda_cost_per_sec = 0.0000333
ec2_cost_per_sec = 0.000003222
lambda_cost_per_sec_ephemeral = 0.00000001545
warmup_cost = 0.0
warmup_time = 0.0
analyse_time = 0
analyse_cost = 0
avg99 = 0.0
avg95 = 0.0
sum_profit_loss = 0.0

def convert_list_to_string(list):
    new_list = []

    for item in list:
        new_list.append(str(item))
    
    return new_list


#Function addapted from lab1
def doRender(tname, values={}):
	if not os.path.isfile( os.path.join(os.getcwd(), 'templates/'+tname) ):
		return render_template('index.htm')
	return render_template(tname, **values) 

#Function addapted from lab3
def getresult(id, operation):
	try:
		host = "qb1ems7lbd.execute-api.us-east-1.amazonaws.com"
		c = http.client.HTTPSConnection(host)
		json= f'{{ "operation": "{operation}" }}'
		c.request("POST", "/default/function_one", json)
		response = c.getresponse() 
		data = response.read().decode('utf-8')
		return( data )
	except IOError:
		print( 'Failed to open ', host )
	return "unusual behaviour of "+str(id)

#Function addapted from lab3
def getresults(parallel, operation):
	with ThreadPoolExecutor() as executor:
		results = executor.map(lambda x: getresult(x, operation), parallel)
	return results

#Function to access the Lambda for the mediate service
def mediate(instances, operation):
    try:
        host = "noxdvndmx2.execute-api.us-east-1.amazonaws.com"
        path = "/default/function_two"

        data = {
        "operation": operation,
        "instances": instances  
        }
        json_data = json.dumps(data) 

        c = http.client.HTTPSConnection(host)
        c.request("POST", path, body=json_data, headers={"Content-Type": "application/json"})

        response = c.getresponse()
        response_data = response.read().decode('utf-8')
        return json.loads(response_data)
    except IOError:
        print( 'Failed to open ', host )
        return "unusual behaviour of "+str(id)
      
# audit api
def mediate_s3(action, body):
    try:
        host = "6u3b0dt986.execute-api.us-east-1.amazonaws.com"
        path = "/default/function_three"

        data = {
        "action": action,
        "data": body  
        }
        json_data = json.dumps(data)  

        c = http.client.HTTPSConnection(host)
        c.request("POST", path, body=json_data, headers={"Content-Type": "application/json"})

        response = c.getresponse()
        response_data = response.read().decode('utf-8')
        return json.loads(response_data)
    except IOError:
        print( 'Failed to open ', host )
        return "unusual behaviour of "+str(id)


# Define the function to make the request to EC2
def make_request(instance_url, data):
    try:
        response = requests.post(instance_url, data=data)
        return response.json()
    except Exception:
         return{
              'warm':'False'
         }
    
# Function to make a request to an EC2 instance
def warm_make_request(instance_url, data, timeout=10):
    try:
        response = requests.post(instance_url, data=data, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        return result
    except requests.exceptions.Timeout:
        return None
    except requests.RequestException as e:
        return None

#Function addapted from lab3
def lambdaAnalyse(id, operation, t, mean, std, shots):
	try:
		host = "qb1ems7lbd.execute-api.us-east-1.amazonaws.com"
		c = http.client.HTTPSConnection(host)
		json= f'{{ "operation": "{operation}","t": "{t}", "mean": "{mean}", "std": "{std}", "shots": "{shots}"}}'
		c.request("POST", "/default/function_one", json)
		response = c.getresponse() 
		data = response.read().decode('utf-8')
		print( data, " from Thread", id )
		return( data )
	except IOError:
		print( 'Failed to open ', host )
	return "unusual behaviour of "+str(id)

#Function addapted from lab3
def getLambdaResults(parallel, operation, t, mean, std, shots):
	with ThreadPoolExecutor() as executor:
		results = executor.map(lambda x: lambdaAnalyse(x, operation, t, mean, std, shots), parallel)
	return results


@app.route('/warmup', methods = ['POST'])
def warmup():
    if request.method == 'POST':
        global scale_serv, warmup_cost, warmup_time, lambda_warmup_results
        data = request.get_json()

        scale_serv['r'] = int(data['r'])
        scale_serv['s'] = data['s']

        if scale_serv['s'] == 'lambda':
            parallel=[value for value in range(scale_serv['r'])]

            start_time = time.perf_counter()

            results = getresults(parallel, "warmup")

            end_time = time.perf_counter()

            warmup_time = end_time - start_time

            warmup_cost = ((warmup_time * lambda_cost_per_sec) * scale_serv['r']) + ((warmup_time * lambda_cost_per_sec_ephemeral) * scale_serv['r'])
    
            results = [json.loads(result) for result in results]
            
            lambda_warmup_results = results

            return jsonify({ 
              "result": "ok"
            })

        elif scale_serv['s'] == 'ec2':

            start_time = time.perf_counter()

            results = mediate(scale_serv['r'], "warmup")

            end_time = time.perf_counter()

            warmup_time = end_time - start_time

            warmup_cost = (warmup_time * lambda_cost_per_sec) + ((ec2_cost_per_sec * 1) * scale_serv['r'])

            return { 
              "result": "ok"
            }
        else:
             return { 
              'statusCode': 400,
              'service': 'error'
             }

@app.route('/scaled_ready', methods = ['GET'])
def scaled_ready():
    if request.method == "GET":
        if scale_serv['s'] == 'lambda':
            count = 0
            for status in lambda_warmup_results:
                if status['status'] == 'warmed up':
                    count += 1

            return jsonify({"warm": "true" if count == scale_serv['r'] else "false"})
        elif scale_serv['s'] == 'ec2':
            try:
                temp = mediate(0, "publicdnsname")

                ec2_url = []

                for x in range(scale_serv['r']):
                    ec2_url.append('http://' + temp['body'][x]['endpoint'] + '/cgi-bin/riskvalues.py')

                data = {
                'mean': 5,
                'std':  2,
                'shots': 1,
                't': 'buy'
                }
                results = []

                with ThreadPoolExecutor(max_workers=len(ec2_url)) as executor:
                    futures = [executor.submit(warm_make_request, url, data) for url in ec2_url]
                    results = [future.result() for future in futures]

                all_warm = all(result is not None for result in results)

                return jsonify({"warm": "true" if all_warm else "false"})
            except Exception as e:
                return {
                    "warm": "false"
                }
        else:

            return{
                "error": "no scalable service defined"
            }

@app.route('/terminate', methods = ['GET'])
def terminate():
    if request.method == "GET":
        global h, d, t, p, profit_loss, var95, var99,warmup_cost, warmup_time, analyse_time, analyse_cost, avg95, avg99, sum_profit_loss

        if scale_serv['s'] == 'lambda':

            h = 0
            d = 0
            t = ""
            p = 0
            profit_loss = []
            var95 = []
            var99 = []
            warmup_cost = 0.0
            warmup_time = 0.0
            analyse_time = 0
            analyse_cost = 0
            avg99 = 0.0
            avg95 = 0.0
            sum_profit_loss = 0.0

            return{
                'result':'ok'
            }
        elif scale_serv['s'] == "ec2":

            h = 0
            d = 0
            t = ""
            p = 0
            profit_loss = []
            var95 = []
            var99 = []
            warmup_cost = 0.0
            warmup_time = 0.0
            analyse_time = 0
            analyse_cost = 0
            avg99 = 0.0
            avg95 = 0.0
            sum_profit_loss = 0.0
            results = mediate(0,"terminate")

            return{
                "result":"ok"
            }
        else:
             return{
                  "error":"not matching"
             }

@app.route('/scaled_terminated', methods = ['GET'])
def scaled_terminated():
    if request.method == 'GET':
        if scale_serv['s'] == 'lambda':

            return jsonify({ 
            "terminated": "true"
            })
        elif scale_serv['s'] == 'ec2':
            results = mediate(0,"checkterminate")

            if(results['statusCode'] == 200):
                return{
                "terminated": "true"
                }

            return{
            "terminated": "false"
            }
        return{
        'error': 400
        }

@app.route('/get_endpoints', methods=['GET'])
def get_endpoints():
     if request.method == 'GET':
          if scale_serv['s'] == 'lambda':
               return jsonify({
                "endpoint": """curl -X POST -d '{"operation": "analyse", "t":"buy", "mean": 20, "std": 10, "shots": 10}' https://qb1ems7lbd.execute-api.us-east-1.amazonaws.com/default/function_one"""
                })
          elif scale_serv['s'] == 'ec2':
               
            results = mediate(0, "publicdnsname")
            dns_list = results['body']
            params = "mean=5.0&std=2.0&shots=1000&t=sell"

            curl_commands = [
            {'endpoint': f'curl -d "{params}" http://{dns["endpoint"]}/cgi-bin/riskvalues.py'}
            for dns in dns_list
            ]

            return jsonify(curl_commands)
          else:
               return{
                    'error': 400
               }


@app.route('/analyse', methods=['POST'])
def analyse():
    if request.method == 'POST':
        global h, d, t, p,var95, var99, profit_loss, analyse_cost, analyse_time, avg95, avg99, sum_profit_loss

        data = request.get_json()

        h = int(data['h'])
        d = data['d']
        t = data['t']
        p = int(data['p'])

        profit_loss = []
        var95 = []
        var99 = []

        yf.pdr_override()
        today = date.today()
        timePast = today - timedelta(days=1095)
        datasets = pdr.get_data_yahoo('MSFT', start=timePast, end=today)

        datasets['buy']=0
        datasets['sell']=0

        for i in range(2, len(datasets)): 
            body = 0.01

            if (datasets.Close[i] - datasets.Open[i]) >= body  \
        and datasets.Close[i] > datasets.Close[i-1]  \
        and (datasets.Close[i-1] - datasets.Open[i-1]) >= body  \
        and datasets.Close[i-1] > datasets.Close[i-2]  \
        and (datasets.Close[i-2] - datasets.Open[i-2]) >= body:
                datasets.at[datasets.index[i], 'buy'] = 1

            if (datasets.Open[i] - datasets.Close[i]) >= body  \
        and datasets.Close[i] < datasets.Close[i-1] \
        and (datasets.Open[i-1] - datasets.Close[i-1]) >= body  \
        and datasets.Close[i-1] < datasets.Close[i-2]  \
        and (datasets.Open[i-2] - datasets.Close[i-2]) >= body:
                datasets.at[datasets.index[i], 'sell'] = 1

        if scale_serv['s'] == 'lambda':
            analyse_time = 0.0
            parallel=[value for value in range(scale_serv['r'])]

            for i in range(h, len(datasets)):
                if datasets[t][i] == 1:
                    mean = datasets.Close[i - h:i].pct_change(1).mean()
                    std = datasets.Close[i - h:i].pct_change(1).std()

                    time_start = time.perf_counter()

                    values = getLambdaResults(parallel, "analyse", t, mean, std, d)
                    
                    time_end = time.perf_counter()

                    analyse_time += (time_end - time_start)

                    results = [json.loads(value) for value in values]

                    temp_var95 = 0.0
                    for x in range(scale_serv['r']):
                        temp_var95 += float(results[x]['var95'])

                    temp_var95 = temp_var95 /scale_serv['r']

                    temp_var99 = 0.0
                    for x in range(scale_serv['r']):
                        temp_var99 += float(results[x]['var99'])

                    temp_var99 = temp_var99 /scale_serv['r']

                    temp_var95 = round(temp_var95,7)
                    temp_var99 = round(temp_var99,7)
                    var95.append(temp_var95)
                    var99.append(temp_var99)

                    if t == 'buy':
                        if i + p < len(datasets):
                            profit_loss_value = datasets.Close[i + p] - datasets.Close[i]
                            profit_loss.append(profit_loss_value)
                    else:
                         if i + p < len(datasets):
                            profit_loss_value =  datasets.Close[i] - datasets.Close[i + p]
                            profit_loss.append(profit_loss_value)

            analyse_cost = ((analyse_time * lambda_cost_per_sec) * scale_serv['r']) + ((analyse_time * lambda_cost_per_sec_ephemeral) * scale_serv['r'])

            avg95 = sum(var95)/len(var95)
            avg99 = sum(var99)/len(var99)
            
            avg95 = round(avg95,7)
            avg99 = round(avg99,7)
           
            sum_profit_loss = sum(profit_loss)

            # Create audit entry
            audit_entry = {
            "s": scale_serv['s'],
            "r": str(scale_serv['r']),
            "h": str(h),
            "d": str(d),
            "t": str(t),
            "p": str(p),
            "profit_loss": str(sum_profit_loss),
            "av95": str(avg95),
            "av99": str(avg99),
            "time": str(analyse_time),
            "cost": str(analyse_cost)
            }

            mediate_s3('save', audit_entry)

            return{
            "result": "ok"
            }
        
        elif scale_serv['s'] == 'ec2':

            temp = mediate(0, "publicdnsname")
            
            ec2_url = []

            for x in range(scale_serv['r']):
                ec2_url.append('http://' + temp['body'][x]['endpoint'] + '/cgi-bin/riskvalues.py')
            
            analyse_time  = 0.0

            for i in range(h, len(datasets)):
                if datasets[t][i] == 1:
                    mean = datasets.Close[i - h:i].pct_change(1).mean()
                    std = datasets.Close[i - h:i].pct_change(1).std()

                    data = {
                    'mean': mean,
                    'std':  std,
                    'shots': d,
                    't': t
                    }
                    results = []

                    time_start = time.perf_counter()

                    with ThreadPoolExecutor(max_workers=len(ec2_url)) as executor:
                        futures = [executor.submit(make_request, url, data) for url in ec2_url]
                        results = [future.result() for future in futures]

                    time_end = time.perf_counter()

                    analyse_time += (time_end - time_start)

                    temp_var95 = 0.0
                    for x in range(scale_serv['r']):
                        temp_var95 += float(results[x]['var95'])

                    temp_var95 = temp_var95 /scale_serv['r']

                    temp_var99 = 0.0
                    for x in range(scale_serv['r']):
                        temp_var99 += float(results[x]['var99'])

                    temp_var99 = temp_var99 /scale_serv['r']

                    temp_var95 = round(temp_var95,7)
                    temp_var99 = round(temp_var99,7)

                    var95.append(temp_var95)
                    var99.append(temp_var99)

                    if t == 'buy':
                        if i + p < len(datasets):
                            profit_loss_value = datasets.Close[i + p] - datasets.Close[i]
                            profit_loss.append(profit_loss_value)
                    else:
                         if i + p < len(datasets):
                            profit_loss_value =  datasets.Close[i] - datasets.Close[i + p]
                            profit_loss.append(profit_loss_value)

            analyse_cost = (analyse_time * ec2_cost_per_sec) * scale_serv['r']

            avg95 = sum(var95)/len(var95)
            avg99 = sum(var99)/len(var99)
            
            avg95 = round(avg95,7)
            avg99 = round(avg99,7)
           
            sum_profit_loss = sum(profit_loss)

            audit_entry = {
            "s": scale_serv['s'],
            "r": str(scale_serv['r']),
            "h": str(h),
            "d": str(d),
            "t": str(t),
            "p": str(p),
            "profit_loss": str(sum_profit_loss),
            "av95": str(avg95),
            "av99": str(avg99),
            "time": str(analyse_time),
            "cost": str(analyse_cost)
            }

            mediate_s3('save', audit_entry)

            return{
            "result": "ok"
            }

        return jsonify({
            'result': 'error'
        })
          
@app.route('/get_sig_vars9599', methods=['GET'])
def get_sig_vars9599():
     if request.method == 'GET':
          return{
               "var95":convert_list_to_string(var95),
               "var99":convert_list_to_string(var99)
          }

@app.route('/get_avg_vars9599', methods=['GET'])
def get_avg_vars9599():
    if request.method == 'GET':
         return{
               "var95":str(avg95),
               "var99":str(avg99)
          }
@app.route('/get_sig_profit_loss', methods=['GET'])
def get_sig_profit_loss():
     if request.method == 'GET':
          return{
               "profit_loss": convert_list_to_string(profit_loss)
          }

@app.route('/get_tot_profit_loss', methods=['GET'])
def get_tot_profit_loss():
     if request.method == 'GET':
          return{
               "profit_loss": str(sum_profit_loss)
          }

@app.route('/get_audit', methods=['GET'])
def get_audit():
     if request.method == 'GET':
        audit_data = mediate_s3('retrieve','[]')
        audit_data = audit_data['body']
        formatted_audit_entries = [
            f'{{s: "{entry["s"]}", r: "{str(entry["r"])}", h: "{str(entry["h"])}", d: "{str(entry["d"])}", t: "{entry["t"]}", p: "{str(entry["p"])}", profit_loss: "{str(entry["profit_loss"])}", av95: "{str(entry["av95"])}", av99: "{str(entry["av99"])}", time: "{str(entry["time"])}", cost: "{str(entry["cost"])}"}}'
            for entry in audit_data
        ]
        formatted_audit = "{\n" + ",\n".join(formatted_audit_entries) + "\n}"
        return Response(formatted_audit, mimetype='application/json')
     
@app.route('/reset', methods=['GET'])
def reset():
    if request.method == 'GET':
        global h, d, t, p, profit_loss, var95, var99, analyse_time, analyse_cost, avg95, avg99, sum_profit_loss

        h = 0
        d = 0
        t = ""
        p = 0
        profit_loss = []
        var95 = []
        var99 = []
        analyse_time = 0.0
        analyse_cost = 0.0
        avg99 = 0.0
        avg95 = 0.0
        sum_profit_loss = 0.0

        return{
             "result": "ok"
        }

@app.route('/get_warmup_cost', methods = ['GET'])
def get_warmup_cost():
     if request.method == 'GET':
          return{
               "billable time" : str(warmup_time),
               "cost" : str(warmup_cost)
          }

@app.route('/get_time_cost', methods = ['GET'])
def get_time_cost():
    if request.method == 'GET':
        return{
            "time": str(analyse_time),
            "cost" : str(analyse_cost)
        }
    
@app.route('/get_chart_url', methods=['get'])
def get_chart_url():
     if request.method == 'GET':
        var_avg_95_list = [avg95 for _ in range(len(var95))]
        var_avg_99_list = [avg99 for _ in range(len(var99))]  
        
        var_95_string = ','.join(str(x) for x in var95)
        var_99_string = ','.join(str(x) for x in var99)
        avg_95_string = ','.join(str(x) for x in var_avg_95_list)
        avg_99_string = ','.join(str(x) for x in var_avg_99_list)
        chart_url = f"http://image-charts.com/chart?cht=lc&chd=a:{var_95_string}|{var_99_string}|{avg_95_string}|{avg_99_string}&chco=fdb45c,27c9c2,FF0000,00FF00&chs=999x999&chdl=var95|var99|avg95|avg99&chxt=y"
        
        return{
             "url": chart_url
        }
 
 # below codes from lab 1                 
@app.route('/cacheavoid/<name>')
def cacheavoid(name):
    if not os.path.isfile( os.path.join(os.getcwd(), 'static/'+name) ):
        return ( 'No such file ' + os.path.join(os.getcwd(), 'static/'+name) )
    f = open ( os.path.join(os.getcwd(), 'static/'+name) )
    contents = f.read()
    f.close()
    return contents

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def mainPage(path):
	return doRender(path)

@app.errorhandler(500)
def server_error(e):
    logging.exception('ERROR!')
    return """
    An  error occurred: <pre>{}</pre>
    """.format(e), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)

