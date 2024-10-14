# Created: 12/06/2019
# Author: Stefan Pantaleev
# Script for post-processing DEM simulations of a static angle of repose test

#Importing libraries
from edempy import Deck
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import os
import os.path
import csv

#Reading in simulation data
for root, dirs, files in os.walk(os.curdir):
    for file in files:
        if file.endswith(".dem"):
            name=file.replace(".dem","")
            print ("-------------------------------------------------------")
            print ("Loading: "+str(name)+".dem")
            print ("-------------------------------------------------------")
            deck=Deck(os.path.join(root,file))

            last_timestep = deck.numTimesteps - 1
        
            #Reading in preferences
    
            if os.path.exists(os.path.join(root,'Static_angle_of_repose_analyst_settings.txt')): 
                with open(os.path.join(root,'Static_angle_of_repose_analyst_settings.txt'), 'r') as file:
                    preferences=file.readlines()
                    sim_end=float(preferences[3])
                    top_rad=float(preferences[5])
                    base_rad=float(preferences[7])
                    bin_size=float(preferences[9])
                    angles=int(preferences[11])
                    report=str(preferences[13])
                    summary=str(preferences[15])
                    plots=str(preferences[17])
                    file.close()
                    settings=True
            else:
                settings=False
                sim_end=0
            #Check if simulation is run to the end
            
            if (sim_end-np.amax(deck.timestepValues)<0.001 and settings==True):
                
                print ("-------------------------------------------------------")
                print ("Processing: "+str(name)+".dem")
                print ("-------------------------------------------------------")
                
                #Finding key timesteps
                
                def find_nearest(array, value):
                    array=np.array(array)
                    timestep = (np.abs(array-value)).argmin()
                    return timestep
                
                fill_end=find_nearest(deck.timestepValues,(sim_end))
                sim_end=find_nearest(deck.timestepValues,(sim_end))
                #Dividing domain into bins
                spacing=np.linspace(top_rad,base_rad,int((base_rad-top_rad)/bin_size))
                theta=np.linspace(0,360,angles)*math.pi/180
                
                #Declarinf arrays
                delta=np.zeros(len(theta))
                index_nonzero=np.zeros(len(spacing))
                SurfaceZ=np.zeros(shape=(len(theta),len(spacing)))
                SurfaceY=np.zeros(shape=(len(theta),len(spacing)))
                SurfaceX=np.zeros(shape=(len(theta),len(spacing)))
                Coord=np.zeros((1,3))
                
                #Getting particle centers
                for n in range(deck.timestep[last_timestep].numTypes):
                    try:
                        Coord=np.append(Coord,deck.timestep[sim_end].particle[n].getPositions(),axis=0)
                    except:
                        continue
                
                Coord = np.delete(Coord, (0), axis=0)
                
                
                if (plots=="Yes\n"):
                    #Set up figure
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(Coord[:,0],Coord[:,1],Coord[:,2],s=0.01,c="y")
                    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                    ax.set_title(str(name))
                
                #Loop through bins and find highest particle
                for i in range(len(theta)):
                    #Find bin centers
                    Grid_x=math.cos(theta[i])*spacing
                    Grid_y=math.sin(theta[i])*spacing
                    #Find surface particles
                    for j in range (len(spacing)):
                        distances=np.sqrt(np.square(Coord[:,0]-Grid_x[j])+np.square(Coord[:,1]-Grid_y[j]))
                        #index_coord=np.where((Coord[:,0]>(Grid_x[j]-bin_size/2)) & (Coord[:,0]<(Grid_x[j]+bin_size/2)) & (Coord[:,1]>(Grid_y[j]-bin_size/2)) & (Coord[:,1]<(Grid_y[j]+bin_size/2)))
                        index_coord=np.where(distances<bin_size/2)
                        surf=Coord[index_coord]
                        #Index zero values
                        if surf.shape[0]>0:
                            Max=np.argmax(surf[:,2])
                            SurfaceX[i][j]=surf[Max,0]
                            SurfaceY[i][j]=surf[Max,1]
                            SurfaceZ[i][j]=surf[Max,2]
                            index_nonzero[j]=j
                        else:
                            index_nonzero[j]=-1
                    #Linear fit to surface particles
                    
                    if spacing[index_nonzero!=-1].shape[0]>2:
                           
                        fit=np.polyfit(spacing[index_nonzero!=-1],SurfaceZ[i][index_nonzero!=-1],1)
                        #Calculating angle of repose and statistics
                        delta[i]=math.atan(abs(fit[0]))*180/math.pi
                    
                        #Plot linear fits
                        if (plots=="Yes\n"):
                            ax.scatter(SurfaceX[i][index_nonzero!=-1],SurfaceY[i][index_nonzero!=-1],SurfaceZ[i][index_nonzero!=-1],s=10,marker='o')
                            ax.plot(Grid_x,Grid_y,(spacing*fit[0]+fit[1]))
                            ax.set_xlabel('X coordinate')
                            ax.set_ylabel('Y coordinate')
                            ax.set_zlabel('Z coordinate')
                
                delta_mean=np.average(delta[delta>0])
                delta_std=np.std(delta[delta>0])
                delta_cov=delta_std/delta_mean*100
                
                #Export figure    
                if (plots=="Yes\n"):
                    #plt.show()
                    fig.savefig(str(name)+'_SAoR.png',dpi=150)
                
                #Reading material parameters and interactions
                        
                Material_Parameters=deck.timestep[last_timestep].materials.getMaterials()
                Interaction_Parameters=deck.timestep[last_timestep].interactions.getInteractions()

                #Writing data to files
                    
                Names = ["Angle of repose (deg)","StDev (deg)","CoV (%)"]    
                Values= [delta_mean,delta_std,delta_cov]
                Units = ["deg","deg","%"]
                Empty_Column=["","",""]
                    
                if (report=="Yes\n"):    
                    with open(str(name) + "_Report" + ".csv", 'w', newline='') as csvFile:
                        writer = csv.writer(csvFile)
                        writer.writerow(["Operational parameters"])
                        writer.writerow(["Bin size (m)","Number of measurements "])
                        writer.writerows(np.column_stack([bin_size,angles]))
                        writer.writerow(["Material parameters"])
                        writer.writerow(["Material","Poisson ratio","Shear modulus (Pa)","Density (kg/m^3)","Work function","Type"])
                        writer.writerows(Material_Parameters)
                        writer.writerow(["Interaction parameters"])
                        writer.writerow(["Interaction","Restitution","Static friction","Rolling friction"])
                        for line in Interaction_Parameters:
                            writer.writerow(line)
                        writer.writerow(["Results"])
                        for line in np.column_stack([Names,Values,Units,Empty_Column]):
                            writer.writerow(line)
                        writer.writerow(["Slice orientation (deg)","Angle of repose (deg)"])
                        for line in np.transpose([theta*180/math.pi, delta]):
                            writer.writerow(line)
                    csvFile.close()
                
                if (summary=="Yes\n"):
                    if os.path.exists("Summary.csv"):
                        with open("Summary.csv", 'a', newline='') as csvFile:
                            writer = csv.writer(csvFile)
                            writer.writerow(np.concatenate((str(name),Values),axis=None))
                            csvFile.close()
                    else:
                        with open("Summary.csv", 'w', newline='') as csvFile:
                            writer = csv.writer(csvFile)
                            writer.writerow(["Simulation", "Angle of repose (deg)","StDev (deg)","CoV (%)"])
                            writer.writerow(np.concatenate((str(name),Values),axis=None))
                            csvFile.close()
    
            else:
                if (settings == False):
                    print ("----------------------------------------------------------------------")
                    print (str(name)+".dem"+" : Settings file not found. Moving to next simulation")
                    print ("----------------------------------------------------------------------")
                else:
                    print ("--------------------------------------------------------------------")
                    print (str(name)+".dem"+" : Simulation unfinished. Moving to next simulation")
                    print ("--------------------------------------------------------------------")
print ("-------------------------------------------------------")
print ("No more simulations in the folder-analysis is complete!")        
print ("-------------------------------------------------------")      
            
            
            
            