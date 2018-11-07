"""
Compute predicted game metrics from user input and return values to be displayed

Load pickled models. Format user input from feat.html and use that input to 
compute predicted downloads and median hours played from regression models.
Return predicted values for user input and predicted values for next three highest
ROI features (importance judged by regression coefficients). Returned values
passed to players.html to dispaly results. Return rendered players.html.

Created by: Kyle Dillon
Last updated: 10-7-18
"""

from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

def label(feature):
    if feature == 'controller':
        return "Support Handheld Controller: "
    elif feature == 'inApp':
        return "In-Game Purchases: "
    elif feature == 'onMac':
        return "Available on Mac: "
    elif feature == 'multi':
        return "Multiplayer: "
    elif feature == 'coop':
        return "Cooperative Play: "
    elif feature == 'mmo':
        return "Massively Multiplayer Online: "
    elif feature == 'isIndie':
        return "Indie Genre: "
    elif feature == 'isAction':
        return "Action Genre: "
    elif feature == 'casual':
        return "Casual Genre: "
    elif feature == 'strat':
        return "Strategy Genre: "
    elif feature == 'rpg':
        return "RPG Genre: "
    elif feature == 'simulation':
        return "Simulation Genre: "
    elif feature == 'sports':
        return "Sports Genre: "
    elif feature == 'racing':
        return "Racing Genre: "
    elif feature == 'price':
        return "Initial Price: "
    else:
        return "blah: "

@app.route('/')
def feat():
   return render_template('feat.html')

@app.route('/result',methods = ['POST', 'GET'])
def players():
    
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    loaded_model2 = pickle.load(open('finalized_model.sav', 'rb'))
    loaded_model3 = pickle.load(open('finalized_model.sav', 'rb'))
    loaded_model4 = pickle.load(open('finalized_model.sav', 'rb'))
#    predX = np.array([])
    predX2 = pd.DataFrame(columns=['feat','select'])
    feats2 = pd.DataFrame(columns=['feature','selected','coef'])
    
    modelhours = pickle.load(open('finalized_model_hours.sav', 'rb'))
    modelhours2 = pickle.load(open('finalized_model_hours.sav', 'rb'))
    modelhours3 = pickle.load(open('finalized_model_hours.sav', 'rb'))
    modelhours4 = pickle.load(open('finalized_model_hours.sav', 'rb'))
#    predX = np.array([])
    predHours = pd.DataFrame(columns=['feat','select'])
    featsHours = pd.DataFrame(columns=['feature','selected','coef'])   
    
    if request.method == 'POST':
        # load the model from disk
     
        result = request.form
        i = 0
#        print(result.items())
        for key, value in result.items():
#            predX = np.append(predX, float(value))
            predX2 = predX2.append({'feat': key, 'select': float(value)}, ignore_index=True)
            predHours = predHours.append({'feat': key, 'select': float(value)}, ignore_index=True)
#            print(key, value, loaded_model.coef_[i])
#           feats = np.append(feats, np.array([key, value, loaded_model.coef_[i]]), axis=0)
            feats2 = feats2.append({'feature': key, 'selected': value, 'coef': loaded_model.coef_[i]}, ignore_index=True)
            featsHours = featsHours.append({'feature': key, 'selected': value, 'coef': modelhours.coef_[i]}, ignore_index=True)
#            features.update({key: np.int(value)})
            i = i+1
            #feats=dict(zip(key, value))
#            otherFeats.update({key: value})

            #if np.int(value) == 0:
            #    otherFeat = np.append(otherFeat, key)
#        print(predX2)
#        predX=predX.reshape(1,-1)
#        print(predX)
#        if pd.isnull(predX2['select'].loc[predX2['feat'] == 'price']):
#            predX2['select'].loc[predX2['feat'] == 'price'] = 9.22
#        print(predX2)
        result = loaded_model.predict(predX2['select'].values.reshape(1,-1))
        resulthours = modelhours.predict(predHours['select'].values.reshape(1,-1))
#        result = loaded_model.predict(predX)
        result = np.int(np.exp(result))
        resulthours = np.int(np.exp(resulthours))
#        feats = np.reshape(feats,(15,3))
        feats3 = feats2
        featshours3 = featsHours

        feats3['coef'] = feats3['coef'].abs()
        feats3 = feats3.sort_values(by = 'selected')
        feats4 = feats3.loc[feats3['selected'] == '0']
        feats4 = feats4.sort_values(by = 'coef', ascending=False)
        
        featshours3['coef'] = featshours3['coef'].abs()
        featshours3 = featshours3.sort_values(by = 'selected')
        featshours4 = featshours3.loc[featshours3['selected'] == '0']
        featshours4 = featshours4.sort_values(by = 'coef', ascending=False)
        
        predX2['select'] = predX2['select'].astype(float)
        predHours['select'] = predHours['select'].astype(float)
        
        predX3 = predX2        
        predX3['select'].loc[predX3['feat'] == feats4.iloc[0]['feature']] = 1
#        print(predX3['select'].values.reshape(1,-1))
        result2 = loaded_model2.predict(predX3['select'].values.reshape(1,-1))
        result2 = np.int(np.exp(result2))
        label2 = predX3['feat'].loc[predX3['feat'] == feats4.iloc[0]['feature']].values
#        print(label2)
        label2 = label(label2)
        
        predhours3 = predHours        
        predhours3['select'].loc[predhours3['feat'] == featshours4.iloc[0]['feature']] = 1
#        print(predX3['select'].values.reshape(1,-1))
        resulthours2 = modelhours2.predict(predhours3['select'].values.reshape(1,-1))
        resulthours2 = np.int(np.exp(resulthours2))
        labelhours2 = predhours3['feat'].loc[predhours3['feat'] == featshours4.iloc[0]['feature']].values
#        print(label2)
        labelhours2 = label(labelhours2)
        
        predX4 = predX2
        predX4['select'].loc[predX4['feat'] == feats4.iloc[0]['feature']] = 0
        predX4['select'].loc[predX4['feat'] == feats4.iloc[1]['feature']] = 1
#        print(predX4['select'].values.reshape(1,-1))
        result3 = loaded_model3.predict(predX4['select'].values.reshape(1,-1))
        result3 = np.int(np.exp(result3))
        label3 = predX4['feat'].loc[predX4['feat'] == feats4.iloc[1]['feature']].values
#        print(label3)
        label3 = label(label3)
        
        predhours4 = predHours
        predhours4['select'].loc[predhours4['feat'] == featshours4.iloc[0]['feature']] = 0
        predhours4['select'].loc[predhours4['feat'] == featshours4.iloc[1]['feature']] = 1
#        print(predX4['select'].values.reshape(1,-1))
        resulthours3 = modelhours3.predict(predhours4['select'].values.reshape(1,-1))
        resulthours3 = np.int(np.exp(resulthours3))
        labelhours3 = predhours4['feat'].loc[predhours4['feat'] == featshours4.iloc[1]['feature']].values
#        print(label3)
        labelhours3 = label(labelhours3)
        
        predX5 = predX2
        predX5['select'].loc[predX5['feat'] == feats4.iloc[1]['feature']] = 0
        predX5['select'].loc[predX5['feat'] == feats4.iloc[2]['feature']] = 1
#        print(predX5['select'].values.reshape(1,-1))
        result4 = loaded_model4.predict(predX5['select'].values.reshape(1,-1))
        result4 = np.int(np.exp(result4))
        label4 = predX5['feat'].loc[predX5['feat'] == feats4.iloc[2]['feature']].values
        label4 = label(label4)
        
        predhours5 = predHours
        predhours5['select'].loc[predhours5['feat'] == featshours4.iloc[1]['feature']] = 0
        predhours5['select'].loc[predhours5['feat'] == featshours4.iloc[2]['feature']] = 1
#        print(predX5['select'].values.reshape(1,-1))
        resulthours4 = modelhours4.predict(predhours5['select'].values.reshape(1,-1))
        resulthours4 = np.int(np.exp(resulthours4))
        labelhours4 = predX5['feat'].loc[predX5['feat'] == featshours4.iloc[2]['feature']].values
        labelhours4 = label(labelhours4)
        
#        predX2['select'].loc[predX2['feat'] == feats4.iloc[2]['feature']] = 1
        
#        print(loaded_model.coef_)
        
#        recs1 = feats[feats[:,1].argsort()]
#        recs2 = recs1[recs1[:,1] == '0']
        
#        for i in range(0,recs2.shape[0]):
#            for j in range (0,recs2.shape[1]):
#                recs2[i,j] = recs2[recs2]
                
#            recs2[]

#        recs2[:,2] = np.absolute(recs2[:,2])
#        recs2 = np.abs(recs2)
#        recs2 = recs2[recs2[:,2].argsort()[::-1]]
#        print(recs2)
#        print(feats.shape)        
#        for i in feats:
#            print(feats[i])
#        print(feats3.shape)
        
        
        
    return render_template("players.html",
                           result = result, 
                           result2 = result2, 
                           result3=result3, 
                           result4=result4, 
                           label2=label2, 
                           label3=label3, 
                           label4=label4, 
                           resulthours=resulthours,
                           resulthours2=resulthours2,
                           resulthours3=resulthours3,
                           resulthours4=resulthours4,
                           labelhours2=labelhours2, 
                           labelhours3=labelhours3, 
                           labelhours4=labelhours4)

if __name__ == '__main__':
   app.run(debug = True)
   
   
   
#predX=np.array([0,1,0,0]);
#predX=predX[:,np.newaxis];
   
   
   
   
   