import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import simpledialog
import matplotlib.patches as patches
import time
from colorama import Fore, Style, init
import torch

device = torch.device("cuda")  # assume GPU per your request

print("")
print(Fore.BLUE + "Input the 2D CAD model" + Style.RESET_ALL)

# Data storage
vertex = []
line = []  # Make sure this is defined globally
circle = []
boundary_coordinates_1 = []
horizontal = []
vertical = []
brk_id = []
brk_id_circle =[]
# Setup figure
fig, ax = plt.subplots()
ax.set_title("Draw polygons (1), circles (2), arcs (3). Press 'q' to quit.")
plt.axis('equal')
plt.grid(True)

# State variables
current_mode = 'polygon'  # 'polygon', 'circle', or 'arc'
current_polygon = []
drawing_polygon = True

# Tkinter root (for dialogs)
root = tk.Tk()
root.withdraw()

def autoscale():
    ax.relim()
    ax.autoscale_view()
    ax.set_aspect('equal', adjustable='datalim')
    fig.canvas.draw()

def draw_polygon(points, style='b-'):
    xs, ys = zip(*points)
    ax.plot(xs, ys, style)
    autoscale()

def draw_circle(center, radius):
    circle = patches.Circle(center, radius, edgecolor='green', facecolor='none', linestyle='--')
    ax.add_patch(circle)
    autoscale()

def draw_arc(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    mid = (p1 + p2) / 2
    radius = np.linalg.norm(p1 - p2) / 2
    theta1 = np.degrees(np.arctan2(p1[1] - mid[1], p1[0] - mid[0]))
    theta2 = np.degrees(np.arctan2(p2[1] - mid[1], p2[0] - mid[0]))
    arc = patches.Arc(mid, 2 * radius, 2 * radius, angle=0, theta1=theta1, theta2=theta2, color='purple')
    ax.add_patch(arc)
    autoscale()

def onkey(event):
    global current_polygon, vertex, current_mode

    if event.key == 'q':
        if current_polygon:
            vertex.append(('polygon', current_polygon.copy()))
        print("Final data:\n", vertex)
        plt.close()

    elif event.key == 'c' and current_mode == 'polygon':
        if len(current_polygon) > 2:
            xs, ys = zip(*[current_polygon[-1], current_polygon[0]])
            ax.plot(xs, ys, 'b--')
            vertex.append(('polygon', current_polygon.copy()))
            current_polygon = []
            autoscale()

    elif event.key == 'escape':
        current_polygon = []
        fig.canvas.draw()

    elif event.key == '1':
        current_mode = 'polygon'
        print("Switched to polygon mode.")

    elif event.key == '2':
        current_mode = 'circle'
        try:
            input_str = simpledialog.askstring("Input Circle", "Enter center_x,center_y,radius (or 'break'):")
            parts = input_str.strip().split()

            circles = []
            for p in parts:
                if p.lower() == "break":
                    circles.append("BREAK")
                    brk_id_circle.append(len(circles) - 1)
                    break  # optional: stop further input
                else:
                    cx, cy, r = map(float, p.strip().split(','))
                    circles.append((cx, cy, r))

            # Clean circles (no BREAK)
            clean_circles = [c for c in circles if c != "BREAK"]

            # Draw them
            for (cx, cy, r) in clean_circles:
                draw_circle((cx, cy), r)

            # Save raw + clean data
            circle.extend(circles)  
            vertex.append(('circle', clean_circles))
            print("Circle data updated:", circle)
            print("BREAK index positions:", brk_id)

        except Exception as e:
            print("Invalid input for circle:", e)

    elif event.key == 'i':
        try:
            input_str = simpledialog.askstring("Manual Input", "Enter points (e.g., 0,0 1,0 1,1 0,1 break):")
            parts = input_str.strip().split()

            pts = []
            

            for i, p in enumerate(parts):
                if p.lower() == "break":
                    pts.append("BREAK")
                    brk_id.append(len(pts) - 1)  # Position of "BREAK" in pts
                    break  # Optional: stop input here
                else:
                    x, y = map(float, p.strip().split(','))
                    pts.append((x, y))

            # Only draw valid points, ignore "BREAK"
            clean_pts = [p for p in pts if p != "BREAK"]
            draw_polygon(clean_pts, style='g-')

            line.extend(pts)  # Keep raw data, including "BREAK"
            vertex.append(('polygon', clean_pts))  # Store clean version
            print("Line data updated:", line)
            print("BREAK index positions:", brk_id)

        except Exception as e:
            print("Invalid input for manual polygon:", e)
# Connect events

fig.canvas.mpl_connect('key_press_event', onkey)

plt.show()
#-------------------------------------------------------------------------Meshing begins-------------------------------------------------#
print("Finding Xmax & Ymax...")
# Filter only (x, y) points
points = [p for p in line if isinstance(p, tuple)]

# Extract x and y separately
x_values = [p[0] for p in points]
y_values = [p[1] for p in points]

# Compute max values
x_max = max(x_values)
y_max = max(y_values)

print("Xmax =", x_max)
print("Ymax =", y_max)
#-----------------------------------------------------------#
L_ref = x_max       # Set L_ref as X_max
def perp_dist(a0_x, a0_y, a1_x, a1_y, g_x, g_y, L_ref):

    X0, Y0 = a0_x, a0_y
    X1, Y1 = a1_x, a1_y
    alpha_g, beta_g = g_x, g_y
    
    numerator   = abs((Y1 - Y0)*alpha_g - (X1 - X0)*beta_g + (X1*Y0) - (Y1*X0))
    denominator = np.sqrt((Y1 - Y0)**2 + (X1 - X0)**2)
    
    r = numerator / denominator
    r = r / L_ref       # scale to non-dimensional units
    r = round(r,6)
    return r    

# def find_repeated():

# Calcuation of mid-boundary coordinates
# Note: We must try to avoid giving geometries smaller than mesh element size
# Example: We want a grid of 129x129 (includes boundaries) so we must input del_h = 10/(129-2) = 10/(127)
# Problem 1: 2,0 10,0 10,2 12,2 12,4 10,4 10,10 2,10 2,8 0,8 0,6 2,6 2,0 break 4,2 8,2 8,8 4,8 4,2 break

del_h = 0.05         #0.05 >> 202x202            #0.0787 >> 129x129             #0.126 >> 81x81
tol = 8
tolerance = 1e-8
space_laps = 1e-4    # (set your input precision)
gpu_tol = 1e-10

cad_st = time.time()
# calculation for line
if (len(line)!=0):
    for i in range(0, len(line)-1, 1):
        X1, X2 = line[i][0], line[i+1][0]
        Y1, Y2 = line[i][1], line[i+1][1]
        
        
        if ((X1 != "B" or Y1 !="R") and (X2 != "B" or Y2 !="R") ):
            
            
            if (abs(X2 - X1) < 1e-5):  # consider vertical line
                # Use np.arange on Y, keep X constant        
                if (Y2 > Y1):
                    for y in np.arange(Y1, Y2, space_laps):
                        boundary_coordinates_1.append((X1, y))
                        vertical.append((X1,y))
                elif (Y2 < Y1):
                    for y in np.arange(Y1, Y2, -(space_laps)):
                        boundary_coordinates_1.append((X1, y))
                        vertical.append((X1,y))


            elif (abs(Y2-Y1) < 1e-5):     # consider horizontal line (simply these lines are not required in the algorithim to define if a point is in or out of domain.
                boundary_coordinates_1.append((X1,Y1))
                if(X2>X1):
                    for x in np.arange(X1,X2,space_laps):
                        horizontal.append((x,Y1))
                    

                elif(X1>X2):
                    boundary_coordinates_1.append((X2,Y2))
                    for x in np.arange(X1,X2,-(space_laps)):
                        horizontal.append((x,Y1))

            else:
                slope = (Y2 - Y1) / (X2 - X1)
                if (Y2 > Y1):
                    print("")
                    for y in np.arange(Y1 , Y2, space_laps ):
                        X = (((y - Y1)/slope) + X1)
                        boundary_coordinates_1.append((X, y))
                        #print("",(X,y))
            
                elif (Y1 > Y2):
                    print("")
                    for y in np.arange(Y1, Y2, -(space_laps)):
                        X = (((y - Y1)/slope) + X1)
                        boundary_coordinates_1.append((X, y))
                        #print("🕊️",(X,y))
                        
                                    
        else:
            pass

if (len(circle)!=0):
    for i in range(0,len(circle),1):
        cx=circle[i][0]
        cy=circle[i][1]
        r= circle[i][2]
        #print("@",cx,cy,r)
        if ((cx != "B" and cy !="R" and r != "E")):   
            #print("🎶",cx,cy,r)
            for i in np.arange((cy-r),(cy+r+space_laps),space_laps):
                io = i - cy
                rounded_io = round(io,8)
                theta=np.arcsin(rounded_io/r)
                r_the=round(theta,8)
                x=cx + r*np.cos(theta)
                y=cy + r*np.sin(theta)
                ###mirror####
                x2=cx-r*np.cos(theta)
                y2=cy+r*np.sin(theta)
                #print( "👍",rounded_io,"",theta,"",x,"",y)
                #print(x,"",y)
                #print(x2,"",y2)
                boundary_coordinates_1.append((x,y))
                boundary_coordinates_1.append((x2,y2))       

# Round the result
rounded_points = [(round(float(x), tol), round(float(y), tol)) for x, y in boundary_coordinates_1]
rounded_horizontal = [(round(float(x), tol), round(float(y), tol)) for x, y in horizontal]
rounded_verticals = [(round(float(x), tol), round(float(y), tol)) for x, y in vertical]
unique_rounded_points = list(set(rounded_points))

cad_ed = time.time()
cad_file_time = cad_ed - cad_st
print(Fore.LIGHTGREEN_EX + "Discretized boundary CAD file is ready !!! " + Style.RESET_ALL)
print(Fore.YELLOW + f"Discretized boundary CAD processing time = {cad_file_time}" + Style.RESET_ALL)


#-----vertex odd-even check-----#
print(Fore.LIGHTMAGENTA_EX + "Starting meshing process..." + Style.RESET_ALL)
mesh_st = time.time()
Even_vertex = []
if (len(line)!=0):
    for ip in range(0,len(brk_id),1):
        
        if (ip>0):
            end_point = np.sum(brk_id[:ip+1]) + ip
            start_point = np.sum([brk_id[:ip]]) + ip
        elif(ip==0):
            start_point = 0
            end_point = brk_id[ip]
        
        vertices = []
        # points before the start point
        vertices.append(line[end_point-3])
        vertices.append(line[end_point-2])
        for ihm in range(start_point,end_point,1):
            vertices.append(line[ihm])

        # point after the end point 
        vertices.append(vertices[3])
        print("😭",vertices)
        
        even_vertex = []
        for i in range(2,len(vertices)-2,1):
            if (abs(vertices[i][1] - vertices[i+1][1]) > 1e-5 or abs(vertices[i][0] - vertices[i+1][0]) > 1e-5):
                if (vertices[i-1][1] < vertices[i][1] and vertices[i+1][1] < vertices[i][1]):
                    even_vertex.append(vertices[i])
                    print(i,vertices[i][0],vertices[i][1])
                elif(vertices[i-1][1] > vertices[i][1] and vertices[i+1][1] > vertices[i][1]):
                    even_vertex.append(vertices[i])
                    print(i,vertices[i][0],vertices[i][1])
                else:
                    print("just passed")
            elif(abs(vertices[i][1] - vertices[i+1][1]) < 1e-5): # 
                if (abs(vertices[i][0] - vertices[i-1][0]) < 1e-5 and abs(vertices[i+1][0] - vertices[i+2][0]) < 1e-5 and ((vertices[i+2][1]  < vertices[i][1] < vertices[i-1][1]) or (vertices[i+2][1]  > vertices[i][1] > vertices[i-1][1]))):
                    even_vertex.append(vertices[i+1])
                    print("V:",vertices[i])
                else:
                    pass
            else:
                print("just passed 2",vertices[i])
                pass

        for im in range(0,len(even_vertex),1):
            unique_rounded_points.append(even_vertex[im]) 
            Even_vertex.append(even_vertex[im])
    
    print("even: ",Even_vertex)
    # time.sleep(900)

print("----------------------------------------")
# print("u: ",unique_rounded_points)
# print("l: ",line)
if(len(circle)!=0):
    for i in range(0,len(circle),1):
        cx=circle[i][0]
        cy=circle[i][1]
        r= circle[i][2]
        if ((cx != "B" and cy !="R" and r != "E")):
            unique_rounded_points.append((cx,cy+r))
            unique_rounded_points.append((cx,cy-r))

s,d = zip(*unique_rounded_points)
plt.scatter(s,d , s=2)
plt.show()

#############################################################################################
Points = []
h = del_h 

for x in np.arange(0,x_max+del_h,h):
    for y in np.arange(0,y_max+del_h,h):
        Points.append((x,y))
print("🏃🏼‍♂️‍➡️: ",len(Points))
# time.sleep(5)
rounded_points_2 = [(round(float(x), tol), round(float(y), tol)) for x, y in Points] # giving same round of to points as given to boundary points 
#points = list(set(rounded_points_2) - (set(rounded_horizontal)|set(rounded_points))) # removing points that lies on boundary and thus are not required to be analyzed. 
boundary_set = set(rounded_horizontal) | set(rounded_points)

points = [p for p in rounded_points_2 if p not in boundary_set]

#------------------------------------ checking of odd-even intersections with respect to x------------------------------------------------#
interior_points_x=[]
for Y in np.arange(0,y_max+del_h,h):

    edge_points_x = []     # Stores data relating to boundary coordinates at particular Y value
    test_point_x = []      # Stores test points from complete space that have same particular Y value

    for i in range(0,len(unique_rounded_points),1):
        if (abs(unique_rounded_points[i][1] - Y) < tolerance):
            edge_points_x.append(unique_rounded_points[i])
        
                
    for j in range(0,len(points),1):
        if(abs(points[j][1] - Y) < tolerance):
            test_point_x.append(points[j])

    for k in range(0,len(test_point_x),1):
        counter_x=[]
        for w in range(0,len(edge_points_x),1):
            if(test_point_x[k][0] < edge_points_x[w][0]):
                counter_x.append("1")

        rem = len(counter_x)
        if (rem % 2 !=0):
            interior_points_x.append(test_point_x[k])
        else:
            pass

mesh_et = time.time()     
mesh_time = mesh_et - mesh_st  
print(Fore.LIGHTGREEN_EX + "Meshing Complete !!!" + Style.RESET_ALL)
print(Fore.YELLOW + f"Mesh processing time = {mesh_time}" + Style.RESET_ALL)

print("WAIT.....")
st = time.time()
unique_rounded_set = set(unique_rounded_points)
filtered_interior_x = [point for point in interior_points_x if point not in unique_rounded_set] # removing boundary points
filtered_interior_x_set = set(filtered_interior_x)  # creating hash-maps for O(1) step time complexity

filtered_exterior = list(set(points) - filtered_interior_x_set)  # All_the_points - interior_points = pureExteriorPoints + boundaryPoints

fill = []
for Y in np.arange(0,y_max+del_h,del_h):
    for u in range(0,len(unique_rounded_points),1):
        if (abs(unique_rounded_points[u][1] - Y) < tolerance):
            fill.append(unique_rounded_points[u])

filtered_exterior_x = list((set(filtered_exterior) | set(rounded_horizontal) |set(fill))) #  pureExteriorPoints + boundaryPoints + HorizontalPoints

filtered_exterior_x_set = set(filtered_exterior_x)
et = time.time()
print("Done.")
print("Time taken = ",et-st)
# print(filtered_exterior_x)
if ((18.2,10) in filtered_exterior_x_set):
    print("yes")
# time.sleep(9000)
#---------------------------------------------------------------ghost-node detection--------------------------------------------------------#
print(Fore.LIGHTMAGENTA_EX + "Starting Edge Detection..." + Style.RESET_ALL)
edge_st = time.time()

ghost_nodes = []
first_interface = []

A = torch.tensor(filtered_interior_x, dtype=torch.float64, device=device)   # shape (N, 2)
B = torch.tensor(filtered_exterior_x, dtype=torch.float64, device=device)   # shape (M, 2)
# print(filtered_interior_x[113],"<<")
for i in range(len(A)):
    A0 = A[i]                          # shape (2,)
    diff = B - A0                      # (M, 2)
    dist = torch.sqrt((diff[:,0])**2 + (diff[:,1])**2)   # GPU
    mask = (torch.abs(dist - del_h) < gpu_tol)              # GPU
    # if (A0[0] == 7.0 and A0[1]==2.5):
    #     print(i,dist,mask)
    idx = torch.where(mask == True)[0]                   # GPU
    # if (A0[0] == 7.0 and A0[1]==2.5):
    #     print(idx)
    id = idx.cpu().numpy()
    if len(id) > 0:
        for j in range(0,len(id),1):
            ghost_nodes.append(filtered_exterior_x[id[j]])
            first_interface.append(filtered_interior_x[i])
    else:
        pass
print("?????",ghost_nodes)
yt = time.time()
print(">>>>>",yt-edge_st)
sorted_ghost_nodes = []
if (len(line)!=0):
    for i in range(0,len(line)-1,1):
        X1,Y1 = line[i][0],line[i][1]       # E
        X2,Y2 = line[i+1][0],line[i+1][1]   # F
        if ((X1 != "B" or Y1 !="R") and (X2 != "B" or Y2 !="R") ):
            print(X1,Y1,"and",X2,Y2)
            dist_EF = np.sqrt((X2-X1)**2 + (Y2-Y1)**2)      # length of EF
            if (abs(X2-X1) < tolerance):
                slope = 0
            else:
                slope = (Y2-Y1)/(X2-X1)
            alpha = np.arctan(slope)
            beta = np.pi - ((np.pi/2 + alpha))
            sub_sorted_ghost_node = []
            for j in range(0,len(ghost_nodes),1):
                X0,Y0 = ghost_nodes[j][0],ghost_nodes[j][1]
                # formula used = (|(y2 - y1)*x0 - (x2-x1)*y0 + (x2*y1 - y2*x1) |) / sqrt ((y2 - y1)**2 + (x2 - x1)**2)
                per_dist = abs(((Y2-Y1)*X0) - ((X2-X1)*Y0) + ((X2*Y1 - Y2*X1)))/(np.sqrt((Y2-Y1)**2 + (X2-X1)**2))
                sin_alpha = abs(per_dist/(np.sin(alpha)))
                sin_beta = abs(per_dist/(np.sin(beta)))
                rounded_sin_alpha = round(sin_alpha,tol)
                rounded_sin_beta = round(sin_beta,tol)
                dist_OE = np.sqrt((X0-X1)**2 + (Y0-Y1)**2)  # length OE
                dist_OF = np.sqrt((X0-X2)**2 + (Y0-Y2)**2)  # length OF
                # print("f: ",X0,Y0,"",per_dist,"",sin_alpha,"",sin_beta)
                # print("f: ",X0,Y0,"",per_dist,"",rounded_sin_alpha,"",rounded_sin_beta)
                if (((rounded_sin_alpha < del_h) or (rounded_sin_beta < del_h)) and ((dist_OE <= dist_EF) and (dist_OF <= dist_EF))):
                    sub_sorted_ghost_node.append((X0,Y0))
                # if (((per_dist < np.sqrt(2*del_h**2))) and ((dist_OE <= dist_EF) and (dist_OF <= dist_EF))):
                #     sub_sorted_ghost_node.append((X0,Y0))               
                else:
                    pass
            if(len(sub_sorted_ghost_node) > 0):
                sorted_ghost_nodes.append(sub_sorted_ghost_node)
            else:
                pass
        else:
            pass        
print("-------")
print("PPPP",sorted_ghost_nodes)
if(len(circle)!=0):
    for i in range(0,len(circle),1):
        cx,cy,r = circle[i][0],circle[i][1],circle[i][2]
        if ((cx != "B" and cy !="R" and r != "E")):
            sub_sorted_ghost_node = []
            for j in range(0,len(ghost_nodes),1):
                x0,y0 = ghost_nodes[j][0],ghost_nodes[j][1]
                dist_OE = np.sqrt((cx-x0)**2 + (cy-y0)**2)
                s = round(abs(r - dist_OE),tol)
                if( abs(cx - x0)< tolerance):
                    # slope = infinity
                    alpha = np.pi/2
                    beta = np.pi - (np.pi/2 + alpha)
                else:    
                    slope = (cy - y0)/(cx - x0)
                    alpha = np.arctan(slope)
                    beta = np.pi - (np.pi/2 + alpha)

                sin_alpha = np.sin(alpha)
                sin_beta = np.sin(beta)
                sx = abs(s/sin_alpha)
                sy = abs(s/sin_beta)
                rounded_sx = round(sx,tol)
                rounded_sy = round(sy,tol)
                        
                print("o: ",x0,y0,"",dist_OE,"",s,"",sx,sy)
                if ( (rounded_sx < del_h) or (rounded_sy < del_h)):
                    sub_sorted_ghost_node.append((x0,y0))
                else:
                    pass
            if(len(sub_sorted_ghost_node) > 0):
                sorted_ghost_nodes.append(sub_sorted_ghost_node)
            else:
                pass    
        else:
            pass
# print("MNJ",first_interface,len(first_interface))
# After analyzing and sorting out the ghost nodes...now we must get the corresponding first interfaces
sorted_first_interface = []
for i in range(0,len(sorted_ghost_nodes),1):
    sub_sorted_first_interface = []
    for j in range(0,len(sorted_ghost_nodes[i]),1):
        X0,Y0 = sorted_ghost_nodes[i][j][0],sorted_ghost_nodes[i][j][1]
        # print(X0,Y0)
        for k in range(0,len(first_interface),1):
            Xf,Yf = first_interface[k][0],first_interface[k][1]
            # if(Xf==7 and Yf==2.5):
            # print(k,Xf,Yf)
            f_dist_g = np.sqrt((X0 - Xf)**2 + (Y0 - Yf)**2)
            if (abs(f_dist_g - del_h ) < tolerance):
                sub_sorted_first_interface.append((Xf,Yf))
                break       # this "break" is important as it limits the finding to only one of the first interface w.r.t the ghost node

            else:
                pass
        # time.sleep(900)
    
    sorted_first_interface.append(sub_sorted_first_interface)

print("++++++++")
print(sorted_ghost_nodes[0])
print(sorted_ghost_nodes[1])
print(sorted_ghost_nodes[2])
print(sorted_ghost_nodes[3])
# print(sorted_ghost_nodes[4])
# print(sorted_ghost_nodes[5])
# print(sorted_ghost_nodes[6])
# print(sorted_ghost_nodes[7])
print("++++++++++")
# time.sleep(900)
#========================================================================================================================================#
#                                                           Mirror Points Setup
#========================================================================================================================================#
mirror = []
mirror_perp_dist = []
if (len(line) !=0 ):
    break_check = 0     # No 'break' passed yet
    for i in range(0, len(sorted_ghost_nodes)+1, 1):        
        if break_check == 0:
            line_index = i
        elif break_check == 1:
            line_index = i+1
        X1, Y1 = line[line_index][0], line[line_index][1]
        X2, Y2 = line[line_index+1][0], line[line_index+1][1]
        print("here: ",X1,Y1,X2,Y2,"",i,line_index,break_check)

        mirror_sub = []
        mirror_perp_dist_sub = []

        if ((X1 != "B" or Y1 !="R") and (X2 != "B" or Y2 !="R") ):
            print("pass: ",X1,Y1,X2,Y2,"",i,line_index,break_check)
            # Handle different line cases
            if abs(X2 - X1) < 1e-10:  # Vertical line
                print("Section_VERTICAL 2: ",X1,Y1,X2,Y2)
                if (break_check == 0):
                    i = i
                elif (break_check == 1):
                    i = i-1
                for j in range(0, len(sorted_ghost_nodes[i]), 1):
                    Xg, Yg = sorted_ghost_nodes[i][j]
                    if (abs(Xg-X1) > tolerance):
                        Xm = 2*X1 - Xg
                        Ym = Yg
                        mirror_sub.append((Xm, Ym))       # formula to be used here Xm = Xg - 2*(Xg-X1)
                        r = perp_dist(X1,Y1,X2,Y2,Xm,Ym,L_ref)
                        mirror_perp_dist_sub.append(r)  # |- perpendicular distnace from the boundary 
                        print("section 1.1: ",X1,Y1,X2,Y2,Xg,Yg,"Edge=",i,"GN_index=",j)
                        print("True_V.0",Xm,Ym)
                    elif(abs(Xg-X1) < tolerance):
                        Xm_p = round((Xg + del_h),tol)      
                        Xm_s = round((Xg - del_h),tol)
                        if ((Xm_p, Yg) in filtered_interior_x_set):
                            mirror_sub.append((Xm_p, Yg))       
                            r = perp_dist(X1,Y1,X2,Y2,Xm_p,Yg,L_ref)
                            mirror_perp_dist_sub.append(r)  # |- perpendicular distnace from the boundary 
                            print("section 1.2: ",X1,Y1,X2,Y2,Xg,Yg,"Edge=",i,"GN_index=",j)
                            print("True_V.1",Xm_p,Yg)
                        elif ((Xm_s, Yg) in  filtered_interior_x_set):
                            mirror_sub.append((Xm_s,Yg))
                            r = perp_dist(X1,Y1,X2,Y2,Xm_s,Yg,L_ref)
                            mirror_perp_dist_sub.append(r)  # |- perpendicular distnace from the boundary 
                            print("section 1.3: ",X1,Y1,X2,Y2,Xg,Yg,"Edge=",i,"GN_index=",j)
                            print("True_V.2",Xm_s,Yg)


                        else:
                            pass
                    
            elif abs(Y2 - Y1) < 1e-10:  # Horizontal line
                print("Section_HORIZONTAL 2: ",X1,Y1,X2,Y2)
                if (break_check == 0):
                    i = i
                elif (break_check == 1):
                    i = i - 1
                for j in range(len(sorted_ghost_nodes[i])):
                    Xg, Yg = sorted_ghost_nodes[i][j]
                    if(abs(Yg - Y1) > tolerance ):
                        print("section 2.1: ",X1,Y1,X2,Y2,Xg,Yg,"Edge=",i,"GN_index=",j)
                        Xm = Xg
                        Ym = 2*Y1 - Yg
                        mirror_sub.append((Xm, Ym))
                        r = perp_dist(X1,Y1,X2,Y2,Xm,Ym,L_ref)
                        mirror_perp_dist_sub.append(r)  # |- perpendicular distnace from the boundary 
                        print("True_H.0",Xm,Ym)
                    elif(abs(Yg - Y1) < tolerance):
                        print("section 2.2: ",X1,Y1,X2,Y2,Xg,Yg,"Edge=",i,"GN_index=",j)
                        Ym_p = round((Yg + del_h),tol)
                        Ym_s = round((Yg - del_h),tol)
                        # print(Ym_p,Ym_s)
                        if ((Xg, Ym_p) in filtered_interior_x_set):
                            mirror_sub.append((Xg, Ym_p))
                            print("True_H.1",Xg,Ym_p)
                            r = perp_dist(X1,Y1,X2,Y2,Xg,Ym_p,L_ref)
                            mirror_perp_dist_sub.append(r)  # |- perpendicular distnace from the boundary 
                        elif ((Xg, Ym_s) in  filtered_interior_x_set):
                            mirror_sub.append((Xg, Ym_s))
                            print("True_H.2",Xg,Ym_s)
                            r = perp_dist(X1,Y1,X2,Y2,Xg,Ym_s,L_ref)
                            mirror_perp_dist_sub.append(r)  # |- perpendicular distnace from the boundary 
                        else:
                            pass
                    
            else:  # Slanted line
                
                target_distance = del_h  
                slope = (Y2 - Y1) / (X2 - X1)
                denominator = slope**2 + 1
                sqrt_den = np.sqrt(denominator)
                if (break_check == 0):
                    i = i
                elif (break_check == 1):
                    i = i - 1
                for j in range(len(sorted_ghost_nodes[i])):
                    Xg, Yg = sorted_ghost_nodes[i][j]
                    print("section 2.3: ",X1,Y1,X2,Y2,Xg,Yg)
                    # print(Xg, Yg) 
                    # Calculate current signed distance
                    # common / sqrt_den is the actual perpendicular distance
                    common = slope * (Xg - X1) - (Yg - Y1)
                    # print(Xg, Yg, common )
                    if abs(common) < tolerance:
                        # POINT IS ON THE LINE: Move by fixed target_distance
                        # We use a negative sign for X and positive for Y to move "inward" 
                        # depending on your domain orientation
                        Xm = Xg - (target_distance * slope) / sqrt_den
                        Ym = Yg + (target_distance) / sqrt_den
                        mirror_perp_dist_sub.append(target_distance/L_ref) # |- Perpendicular distance (expetional case)
                    else:
                        # POINT IS NOT ON THE LINE: Use standard reflection
                        Xm = Xg - (2 * slope * common) / denominator
                        Ym = Yg + (2 * common) / denominator
                        r = perp_dist(X1,Y1,X2,Y2,Xm,Ym,L_ref)
                        mirror_perp_dist_sub.append(r)  # |- Perpendicular distance

                    
                    mirror_sub.append((Xm, Ym))
        else:
            break_check = 1    # 'break'passed
            print("HI ???")
        
        print("mirror_sub_id = ",i," ",mirror_sub)
        if (len(mirror_sub) > 0):
            mirror.append(mirror_sub)
            mirror_perp_dist.append(mirror_perp_dist_sub)
# time.sleep(900)
print(len(sorted_ghost_nodes[2]))
print("LENGTH = ",len(mirror))
print(mirror[0])
print(mirror[1])
print(mirror[2])
print(mirror[3])
# print(mirror[4])
# print(mirror[5])
# print(mirror[6])
# print(mirror[7])
# print(mirror[8])
# print(filtered_interior_x_set)
# time.sleep(900)
#========================================================================================================================================#
#                                                           Interpolation Points Setup
#========================================================================================================================================#
interpolation_points = []
global_normal_dist = []
break_check = 0
# detecting points for bilinear interpolation on mirror point
for i in range(0,len(sorted_ghost_nodes)+1,1):

    if (break_check == 0):
        line_index = i
    elif (break_check == 1):
        line_index = i+1

    X0,Y0 = line[line_index][0],line[line_index][1]
    X1,Y1 = line[line_index+1][0],line[line_index+1][1]

    if ((X0 != "B" or Y0 !="R") and (X1 != "B" or Y1 !="R") ):

        sub_interpolation_points = []
        sub_normal_dist = []

        if (break_check == 0):
            i = i
        elif (break_check == 1):
            i = i - 1

        for j in range(0,len(sorted_ghost_nodes[i]),1):
            coords = []
            normal_dist = []
            Xg,Yg = sorted_ghost_nodes[i][j][0],sorted_ghost_nodes[i][j][1]
            Xm,Ym = mirror[i][j][0],mirror[i][j][1] 
            Max_limit = 3   # maximum size for interpolation stencil
            if True:
                # Y ---
                alpha_y_1 = Yg - del_h

                alpha_y_2 = Yg - 2*del_h

                alpha_y_3 = Yg - 3*del_h

                alpha_y_1 = round(alpha_y_1,tol)

                alpha_y_2 = round(alpha_y_2,tol)

                alpha_y_3 = round(alpha_y_3,tol)
                # if ((Yg - 5.0) < tolerance and (Xg - 5.2) < tolerance):
                # if i - 4 < tolerance:
                    # print(">LOP) ",alpha_y_1,alpha_y_2,alpha_y_3,Xg,Yg)
                    # time.sleep(1)
                if ((Xg, alpha_y_1) in filtered_interior_x_set) and (len(coords) < Max_limit):  
                    r = perp_dist(X0,Y0,X1,Y1,Xg,alpha_y_1,L_ref)
                    if (r > tolerance)  and (r not in normal_dist):
                        normal_dist.append(r)
                        coords.append((Xg,alpha_y_1))
                        print("1> ",alpha_y_1,Xg,Yg,"Y--")

                if((Xg, alpha_y_2) in filtered_interior_x_set) and (len(coords) < Max_limit):
                    r = perp_dist(X0,Y0,X1,Y1,Xg,alpha_y_2,L_ref)
                    if (r > tolerance)  and (r not in normal_dist):
                        normal_dist.append(r)
                        coords.append((Xg,alpha_y_2))
                        print("2> ",alpha_y_2,Xg,Yg,"Y--")

                if((Xg, alpha_y_3) in filtered_interior_x_set) and (len(coords) < Max_limit):
                    r = perp_dist(X0,Y0,X1,Y1,Xg,alpha_y_3,L_ref)
                    if (r > tolerance)  and (r not in normal_dist):
                        normal_dist.append(r)
                        coords.append((Xg,alpha_y_3))
                        print("3> ",alpha_y_3,Xg,Yg,"Y--")

                # X +++
                alpha_x_1 = Xg + del_h

                alpha_x_2 = Xg + 2*del_h

                alpha_x_3 = Xg + 3*del_h

                alpha_x_1 = round(alpha_x_1,tol)

                alpha_x_2 = round(alpha_x_2,tol)

                alpha_x_3 = round(alpha_x_3,tol)

                if ((alpha_x_1, Yg) in filtered_interior_x_set) and (len(coords) < Max_limit):
                    r = perp_dist(X0,Y0,X1,Y1,alpha_x_1,Yg,L_ref)
                    if (r > tolerance)  and (r not in normal_dist): 
                        normal_dist.append(r)
                        coords.append((alpha_x_1,Yg))
                        print("1> ",alpha_x_1,Xg,Yg,"X++")

                if ((alpha_x_2, Yg) in filtered_interior_x_set) and (len(coords) < Max_limit):
                    r = perp_dist(X0,Y0,X1,Y1,alpha_x_2,Yg,L_ref)
                    if (r > tolerance)  and (r not in normal_dist):
                        normal_dist.append(r)
                        coords.append((alpha_x_2,Yg))
                        print("2> ",alpha_x_2,Xg,Yg,"X++")

                if((alpha_x_3, Yg) in filtered_interior_x_set) and (len(coords) < Max_limit):
                    r = perp_dist(X0,Y0,X1,Y1,alpha_x_3,Yg,L_ref)
                    if (r > tolerance)  and (r not in normal_dist):
                        normal_dist.append(r)
                        coords.append((alpha_x_3,Yg))
                        print("3> ",alpha_x_3,Xg,Yg,"X++")
                # X ---
                alpha_x_1 = Xg - del_h

                alpha_x_2 = Xg - 2*del_h

                alpha_x_3 = Xg - 3*del_h  

                alpha_x_1 = round(alpha_x_1,tol)

                alpha_x_2 = round(alpha_x_2,tol)

                alpha_x_3 = round(alpha_x_3,tol)

                if ((alpha_x_1, Yg) in filtered_interior_x_set) and (len(coords) < Max_limit):    
                    r = perp_dist(X0,Y0,X1,Y1,alpha_x_1,Yg,L_ref)
                    if (r > tolerance)  and (r not in normal_dist):
                        normal_dist.append(r)
                        coords.append((alpha_x_1,Yg))
                        print("1> ",alpha_x_1,Xg,Yg,"X--")

                if((alpha_x_2, Yg) in filtered_interior_x_set) and (len(coords) < Max_limit):
                    r = perp_dist(X0,Y0,X1,Y1,alpha_x_2,Yg,L_ref)
                    if (r > tolerance)  and (r not in normal_dist):
                        normal_dist.append(r)
                        coords.append((alpha_x_2,Yg))
                        print("2> ",alpha_x_2,Xg,Yg,"X--")

                if((alpha_x_3, Yg) in filtered_interior_x_set) and (len(coords) < Max_limit):
                    r = perp_dist(X0,Y0,X1,Y1,alpha_x_3,Yg,L_ref)
                    if (r > tolerance)  and (r not in normal_dist):
                        normal_dist.append(r)
                        coords.append((alpha_x_3,Yg))
                        print("3> ",alpha_x_3,Xg,Yg,"X--")
    
                # Y +++
                alpha_y_1 = Yg + del_h

                alpha_y_2 = Yg + 2*del_h

                alpha_y_3 = Yg + 3*del_h

                alpha_y_1 = round(alpha_y_1,tol)

                alpha_y_2 = round(alpha_y_2,tol)

                alpha_y_3 = round(alpha_y_3,tol)

                if ((Xg, alpha_y_1) in filtered_interior_x_set) and (len(coords) < Max_limit):  
                    r = perp_dist(X0,Y0,X1,Y1,Xg,alpha_y_1,L_ref)
                    if (r > tolerance)  and (r not in normal_dist):
                        normal_dist.append(r)
                        coords.append((Xg,alpha_y_1))
                        print("1> ",alpha_y_1,Xg,Yg,"Y++")

                if ((Xg, alpha_y_2) in filtered_interior_x_set) and (len(coords) < Max_limit):
                    r = perp_dist(X0,Y0,X1,Y1,Xg,alpha_y_2,L_ref)
                    if (r > tolerance)  and (r not in normal_dist): 
                        normal_dist.append(r)
                        coords.append((Xg,alpha_y_2))
                        print("2> ",alpha_y_2,Xg,Yg,"Y++")

                if((Xg, alpha_y_3) in filtered_interior_x_set) and (len(coords) < Max_limit):
                    r = perp_dist(X0,Y0,X1,Y1,Xg,alpha_y_3,L_ref)
                    if (r > tolerance)  and (r not in normal_dist):
                        normal_dist.append(r)
                        coords.append((Xg,alpha_y_3))
                        print("3> ",alpha_y_3,Xg,Yg,"Y++")

                print("id = ",i,"Edge = ",X0,Y0,X1,Y1,"GN_index = ",j,Xg,Yg)
                print(coords)
                print(normal_dist)
            # if (len(coords) > 0):
                sub_interpolation_points.append(coords)
                sub_normal_dist.append(normal_dist)

                # print("sub: ",sub_interpolation_points)
        interpolation_points.append(sub_interpolation_points)   
        global_normal_dist.append(sub_normal_dist)
    else:
        break_check = 1
        pass

    # print("Interpol: ",interpolation_points)

# print(interpolation_points[0][7])
# print(">>",global_normal_dist[0][70])
print("*******")
gnc = []
for i in range (0 ,len(sorted_ghost_nodes), 1):
    l = len(sorted_ghost_nodes[i])
    gnc.append(l)

total_number_of_ghost_nodes = np.sum(gnc)
print("**********")


# Building second_interface
second_interface = set(filtered_interior_x) - set(first_interface)

    
edge_et = time.time()
edge_time = edge_et - edge_st
print(Fore.LIGHTGREEN_EX + "Edge detection Complete !!!" + Style.RESET_ALL)
print(Fore.YELLOW + f"Total time to identify edges = {edge_time}" + Style.RESET_ALL)

print("")
print("")
#print(ghost_nodes)
print(Fore.BLUE + "Final mesh !!!" + Style.RESET_ALL)
print(f"total nodes: {len(filtered_interior_x)}")
print("")

print(sorted_ghost_nodes[0])
print(first_interface)



# Unzip coordinates for plotting
x, y = zip(*unique_rounded_points)
if (len(rounded_horizontal) !=0):
    g, h = zip(*rounded_horizontal)
c, d = zip(*points)
a, b = zip(*interior_points_x)
yama,lama = zip(*ghost_nodes)
vv = sorted_ghost_nodes[0]
eera,meera = zip(*vv)
vvc = sorted_first_interface[0]
dxc,vfc = zip(*vvc)
miros = mirror[0]
mxi,myi = zip(*miros)
xe,ye = zip(*filtered_exterior_x)
ioj,lop = zip(*second_interface)


# Plotting
plt.scatter(x, y, color='red', s=1, label='Actual boundaries')       # Red markers
if (len(rounded_horizontal) !=0):
    plt.scatter(g, h, color='red', s=1)
# plt.scatter(c, d, color ="blue", s= 5)
plt.scatter(a, b, color = "black", s = 5)       # x-marker
# plt.scatter(g, h, color = "black", s = 5)       # y-marker

plt.scatter(yama,lama,color = 'green', s = 10)
plt.scatter(eera,meera,color = 'blue', s = 10)
plt.scatter(dxc,vfc, color = "#E0E700", s = 5)
plt.scatter(ioj,lop, color = "#BE0095", s =5)
# plt.scatter(mxi,myi,color = "#823A3A",s = 6)

# plt.scatter(xe,ye, s =7, color = 'blue')
# plt.scatter(xf,yf, s =7, color =  "#AF5858")
# plt.scatter(xipn,yipn, s =7, color = "#A60008")

# plt.scatter(x_even,y_even, color = "yellow",s=0.4)
# Annotations and labels
# plt.title("Polygon Visualization with Red Markers", fontsize=14)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend()
plt.axis('equal')  # Fix aspect ratio
plt.show()


nx = int((x_max+del_h)/del_h) + 1
ny = int((y_max+del_h)/del_h) + 1

print("nx, ny = ",nx,ny)

path1 = r"D:/numerical computation/geometry meshing/Meshes/RAX_1"
path2 = r"D:/numerical computation/geometry meshing/Meshes/GAX_1"

# np.savez(path1, array1=filtered_interior_x)
# sorted_ghost_nodes = np.array(sorted_ghost_nodes, dtype=object)
# sorted_first_interface = np.array(sorted_first_interface, dtype=object)
# np.savez(path2, array1=sorted_ghost_nodes, array2 = sorted_first_interface)

# === Convert to numpy object arrays (important!) ===
sorted_ghost_nodes = np.array(sorted_ghost_nodes, dtype=object)
sorted_first_interface = np.array(sorted_first_interface, dtype=object)
# interpolation_points = np.array(interpolation_points, dtype=object)
mirror_array = np.array(mirror, dtype=object)
second_interface = np.array(second_interface, dtype=object)

# === Save paths ===
path1 = r"D:/numerical computation/geometry meshing/Meshes/RAX_1.npz"
path2 = r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz"

# # # === Save .npz files === 
np.savez(path1, array1=filtered_interior_x, del_h = del_h, nx = nx, ny = ny, total_number_of_ghost_nodes = total_number_of_ghost_nodes)
np.savez(
    path2,
    array1=sorted_ghost_nodes,
    array2=sorted_first_interface,
    array3=interpolation_points,
    array4=mirror_array,
    array5=unique_rounded_points,
    array6=rounded_horizontal,
    array7=sorted_first_interface,
    array8=second_interface,
    array9=first_interface,
    array10=mirror,
    array11=mirror_perp_dist,
    array12=interpolation_points,
    array13=global_normal_dist
)


print("===== Perpendicular_distance ======")
print(global_normal_dist)
print("====mirror point===")
print(mirror)
print(mirror_perp_dist)
print("✅ Geometry data saved successfully!")

# print(">>",global_normal_dist[0][70])
print(filtered_interior_x)
print("mc",mirror)
print("iL: ",interpolation_points)
print(">>>",sorted_ghost_nodes)
