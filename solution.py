import csv
import math
import numpy
from csv import DictReader
from scipy.stats import lognorm, beta
from matplotlib import pyplot as plt

def validation_csvfile():
    with open("Risk Modelling Expert Data Capture.csv", 'r') as file:
        csvreader = csv.DictReader(file)
        csvrows=[]
        csverrors=[]
        notnum=[]
        count=1
        for row in csvreader:
            errors=[]
            for key,value in row.items():
                if(value.isnumeric()):
                    if (int(value)):
                        row[key]= int(value)
                    else:
                        row[key]=float(value)
                else:
                    if(key!='Title'):
                        notnum.append(key)

                if(value== 'null' or value=='' or value==' '):
                    errors.append("missing values ")
            if not ((isinstance(row['prob_min'],int) or  isinstance(row['prob_min'],float))
            and (isinstance(row['prob_most'],int) or  isinstance(row['prob_most'],float))
            and (isinstance(row['prob_max'],int) or  isinstance(row['prob_max'],float))
            and (isinstance(row['lb_loss'],int) or  isinstance(row['lb_loss'],float))
            and (isinstance(row['ub_loss'],int) or  isinstance(row['ub_loss'],float))):
                errors.append("invalid datatypes")
            if('prob_min' not in notnum and 'prob_most' not in notnum and 'prob_max' not in notnum):
                if not (0 <= row['prob_min'] <= row['prob_most'] <= row['prob_max']):
                    errors.append("invalid range prob_min, prob_most and prob_max")
            if('lb_loss' not in notnum and 'ub_loss' not in notnum ):
                if not (0 <= row['lb_loss'] < row['ub_loss']):
                    errors.append("invalid range lb_loss and ub_loss")
        
            if(len(errors)==0):
                row['id']=count
                csvrows.append(row)
            else:
                row['errors']= errors
                row['id']=count
                csverrors.append(row)
            count+=1
        return csvrows,csverrors
    
def distribution_model(csvrows):
    for item in csvrows:
        alpha= 1+4*((item['prob_most']-item['prob_min'])/(item['prob_max']-item['prob_min']))
        bet= 1+4*((item['prob_max']-item['prob_most'])/(item['prob_max']-item['prob_min']))
        pertvalue= beta.rvs(alpha, bet, loc=item['prob_most'], scale=item['prob_max']-item['prob_min'])
        pertval.append(pertvalue)
        item['pmean']= (item['prob_min']+4*item['prob_most']+item['prob_max'])/6
        item['pmedian']= (item['prob_min']+6*item['prob_most']+item['prob_max'])/8
        item['pmode']=item['prob_most']
        item['pstdev']=math.sqrt(((item['pmean']-item['prob_min'])*(item['prob_max']-item['pmean']))/7)
        item['pale']= item['prob_most']*item['pmean']
        pertdistribution.append(item)
        item['lmu']=((math.log(item['ub_loss'])) + (math.log(item['lb_loss'])))/2
        item['lsi']=((math.log(item['ub_loss'])) - (math.log(item['lb_loss'])))/3.29
        logvalue=lognorm(s=item['lsi'], scale=math.exp(item['lmu']))
        logval.append(logvalue)
        item['lmean']= math.e ** (item['lmu']+ ((item['lsi']**2)/2))
        item['lmedian']= math.e **(item['lmu'])
        item['lmode']=math.e **(item['lmu']-(item['lsi']**2))
        lvariance = (math.exp(item['lsi']*2) - 1)*math.exp(2*item['lmu'] + item['lsi']*2)
        item['lstdev'] = math.sqrt(lvariance)
        item['lale']= item['prob_most']*item['lmean']
        logdistribution.append(item)
        pert_log_dist.append(item)
    return pert_log_dist
def monte_carlo_simulation(risk_event,risk_losses,pert_log_dist):
    for year in range(5):
        loss_sum=[]
        loss_event=[]
        for risk in pert_log_dist :
            loss=0
            alpha= 1+4*((risk['prob_most']-risk['prob_min'])/(risk['prob_max']-risk['prob_min']))
            bet= 1+4*((risk['prob_max']-risk['prob_most'])/(risk['prob_max']-risk['prob_min']))
            random_pert=beta.rvs(alpha,bet,loc=risk['prob_most'],scale=risk['prob_max']-risk['prob_most'])
            for pert in range(1, int(random_pert)+1):
              loss +=lognorm.rvs(s=(risk['lsi']), loc=risk['lb_loss'],scale=risk['ub_loss']-risk['lb_loss'])
            loss_sum.append(loss)
            loss_event.append(random_pert)
            if(len(risk_event[risk['Title']])>0 and len(risk_losses[risk['Title']])>0):
              risk_event[risk['Title']] +=[random_pert]
              risk_losses[risk['Title']] +=[loss]
            else:
              risk_event[risk['Title']] =[random_pert]
              risk_losses[risk['Title']] =[loss]
        print(f"average no of events per year {numpy.average(loss_event)}")
        print(f"min no of events per year  {numpy.min(loss_event)}")
        print(f"Max no of events per year  {numpy.max(loss_event)}")
        print(f"Average Loss in a year {numpy.average(loss_sum)}")
        print(f"Min loss in a year {numpy.min(loss_sum)}")
        print(f"Max loss in a year {numpy.max(loss_sum)}")

    losses = numpy.array([numpy.percentile((loss_sum), x) for x in range(1, 100, 1)])
    percentiles = numpy.array([float(100 - x) / 100.0 for x in range(1, 100, 1)])
    print(f"There is a {percentiles[75] * 100}% of losses exceeding {losses[75]:.2f} or more")
    print(f"There is a {percentiles[50] * 100}% of losses exceeding {losses[50]:.2f} or more")
    print(f"There is a {percentiles[25] * 100}% of losses exceeding {losses[25]:.2f} or more")
    return risk_event,risk_losses,losses,percentiles,loss_sum

def risk_event_loss(risk_event,risk_losses):
    for key,value in risk_event.items():
        print(key)
        print(f"total no of times risk event occurred {numpy.sum(value)}")
        print(f"mean Loss {numpy.average(risk_losses[key])}")
        print(f"Min loss {numpy.min(risk_losses[key])}")
        print(f"Max loss {numpy.max(risk_losses[key])}")

def data_visualization(losses,percentiles,loss_sum):
    plt.subplot(1,2,1)
    plt.plot(losses, percentiles)
    title="Aggregated Loss Exceedance"
    plt.title(title)
    plt.xlabel ('losses')
    plt.ylabel ('percentiles')
    
    plt.subplot(1,2,2)
    plt.title("Loss scatter visualization")
    x= numpy.linspace(0,len(loss_sum),len(loss_sum))
    y= x+ 10 * numpy.random.randn(len (loss_sum))
    plt.xlabel('losses')
    plt.scatter(x,y)
    plt.show()
            
if __name__=="__main__":
    
    csvrows,csverrors= validation_csvfile()
    for item in csverrors:
        print(f"{'line_number':<20}| {'errors':>15}")
        print(f"{'-'*21}+{'-'*17}+{'-'*17}")
        print (f"Id: {item['id']}, Errors: {item['errors']}, values: {item['Title']}, {item['prob_min']},{item['prob_most']},{item['prob_max']},{item['lb_loss']},{item['ub_loss']} ")
    pertdistribution=[]
    pertval=[]
    logdistribution=[]
    logval=[]
    pert_log_dist=[]
    alpha=0
    bet=0
    pert_log=distribution_model(csvrows)
    risk_event={}
    risk_losses={}
    for val in csvrows:
        risk_event[val['Title']]= []
        risk_losses[val['Title']] =[]
    risk_event,risk_loss,loss,percentile,loss_sum=monte_carlo_simulation(risk_event,risk_losses,pert_log)
    risk_event_loss(risk_event,risk_loss)
    data_visualization(loss,percentile,loss_sum)