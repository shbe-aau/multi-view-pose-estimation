import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
#from collections import defaultdict
import resource, sys

def shared_points(cell1, cell2):
    shared = []
    for p in cell1:
        if p in cell2:
            shared.append(p)
    return shared

def plot_boundaries(regions, neighbours, sVoronoi, ax):
    # plot the boundary of each region by finding a cell on the outer border and folloing the edge until it loops (assumes no hole topology)
    print(regions[0])
    print(neighbours[0])
    print(sVoronoi.regions[0])
    print(sVoronoi.vertices[0])
    p1 = sVoronoi.vertices[sVoronoi.regions[50][0]]
    p2 = sVoronoi.vertices[sVoronoi.regions[50][1]]
    p3 = sVoronoi.vertices[sVoronoi.regions[50][2]]

    #ax.scatter(p1[0]*1.1, p1[1]*1.1, p1[2]*1.1, c='magenta', s=10)
    #ax.scatter(p2[0]*1.1, p2[1]*1.1, p2[2]*1.1, c='magenta', s=10)
    #ax.scatter(p3[0]*1.1, p3[1]*1.1, p3[2]*1.1, c='magenta', s=10)

    for region in regions:
        border = []
        bx = []
        by = []
        bz = []
        set = regions[region]['region_set']
        # find member of set with a non-same-set neightbour
        start = None
        next = None
        current = None
        for index in set:
            cell = sVoronoi.regions[index]
            neighs = neighbours[index]
            for neigh in neighs:
                if neigh not in set:
                    #a = (sVoronoi.points[index]*1.1)
                    #ax.scatter(a[0], a[1], a[2], c='magenta', s=10)
                    pts = shared_points(cell, sVoronoi.regions[neigh])
                    start = pts[0]
                    next = pts[1]
                    current = index
                    border.append(index)
                    bx = [sVoronoi.vertices[pts[0]][0]*1.1, sVoronoi.vertices[pts[1]][0]*1.1]
                    by = [sVoronoi.vertices[pts[0]][1]*1.1, sVoronoi.vertices[pts[1]][1]*1.1]
                    bz = [sVoronoi.vertices[pts[0]][2]*1.1, sVoronoi.vertices[pts[1]][2]*1.1]
                    break
        i = 0
        print()
        while next is not start:
            i += 1
            #print("{} {}".format(next, start))
            #cell = sVoronoi.regions[current]
            neighs = neighbours[current]
            # find which in-neighbour shares point "next"
            for neigh in neighs:
                if neigh in set and next in sVoronoi.regions[neigh]:# and (neigh is start or neigh not in border):
                    cell = sVoronoi.regions[neigh]
                    neighs2 = neighbours[neigh]
                    for neigh2 in neighs2:
                            if neigh2 not in set:
                                border.append(neigh)
                                #print("{} {}".format(current,neighs))
                                #print("{} {}".format(next,start))
                                #print(border)
                                current = neigh
                                pts = shared_points(cell, sVoronoi.regions[neigh2])
                                if next not in pts:
                                    #print("Error, missing point in neightbour")
                                    continue
                                for p in pts:
                                    if p is not next:
                                        next = p
                                bx.append(sVoronoi.vertices[p][0]*1.1)
                                by.append(sVoronoi.vertices[p][1]*1.1)
                                bz.append(sVoronoi.vertices[p][2]*1.1)
                                ax.scatter(bx, by, bz, c='magenta', s=2)
                                plt.show()
                                continue
            break
        print(border)
        ax.scatter(bx, by, bz, c='magenta', s=2)


# recursively
def steepest_region_for(current, neighbours, losses, steepest, found=set(), current_path=set(), regions={}):
    current_path |= {current}

    if current in found:
        for region in regions:
            if current in regions[region]['region_set']:
                regions[region]['region_set'] |= current_path
                found |= {current}
                return found, regions, steepest

    found |= {current}

    # global index of neighbour with the lowest loss
    index = neighbours[current][np.argmin(losses[neighbours[current]])]
    # if current har smaller loss than than lowest neighbour this it the minima, and has not been found before this
    if losses[index] > losses[current] or (losses[index] >= losses[current] and index > current):
        regions[current] = {'index': current, 'min loss': losses[current], 'region_set': current_path}
        return found, regions, steepest
    steepest[current] = index
    return steepest_region_for(index, neighbours, losses, steepest, found, current_path, regions)

def steepest_region(neighbours, losses):
    resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
    sys.setrecursionlimit(10**6)

    steepest = [-1]*len(losses)

    found = set()
    regions = {}
    for i in range(len(neighbours)):
        if i not in found:
            found_temp, regions_temp, steepest = steepest_region_for(i, neighbours, losses, steepest, found=found, current_path=set(), regions=regions)
            found |= found_temp
            regions.update(regions_temp)
    return regions, steepest

def voronoi_plot(points_in, losses):
    from matplotlib import colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    from scipy.spatial import SphericalVoronoi
    from mpl_toolkits.mplot3d import proj3d
    # get input points in correct format
    cart = [point['cartesian'] for point in points_in]
    points = np.array(cart)
    center = np.array([0, 0, 0])
    radius = 1
    # calculate spherical Voronoi diagram
    sv = SphericalVoronoi(points, radius, center)
    # sort vertices (optional, helpful for plotting)
    sv.sort_vertices_of_regions()
    # generate plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot the unit sphere for reference (optional)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='y', alpha=0.1)

    # normalize and map losses to colormap
    mi = min(losses)
    ma = max(losses)

    norm = matplotlib.colors.Normalize(vmin=mi, vmax=ma, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)

    loss_color = [mapper.to_rgba(l) for l in losses]

    # indicate Voronoi regions (as Euclidean polygons)
    for i in range(len(sv.regions)):
        region = sv.regions[i]
        random_color = colors.rgb2hex(loss_color[i])
        polygon = Poly3DCollection([sv.vertices[region]], alpha=1.0)
        polygon.set_color(random_color)
        ax.add_collection3d(polygon)

    # determine neighbours
    neighbours = []
    for i in range(len(sv.regions)):
        vertices = sv.regions[i]
        neighbours_for_i = []
        for j in range(len(sv.regions)):
            neigh = False
            shared = 0
            vert2 = sv.regions[j]
            for vert in vertices:
                if vert in vert2:
                    shared += 1
                    if shared >= 2:
                        neigh = True
                        break
            if neigh and i != j:
                neighbours_for_i.append(j)
        neighbours.append(neighbours_for_i)

    regions, steepest_neighbour = steepest_region(neighbours, losses)
    #plot_boundaries(regions, neighbours, sv, ax)

    temp = []
    for i in range(len(neighbours)):
        point1 = sv.points[i]
        point2 = points[steepest_neighbour[i]]
        temp.append(point2-point1)

    x1, y1, z1 = zip(*sv.points)
    dx, dy, dz = zip(*temp)

    col = [[0, 0, 0, 0.5] for i in range(len(neighbours))]
    #ax.quiver(x1, y1, z1, dx, dy, dz, length=0.02, normalize=True, colors=col) # uncomment this get directional arrows as well

    num_sets = len(regions)
    colors = cm.rainbow(np.linspace(0, 1, num_sets))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='y', alpha=0.1)
    # indicate Voronoi regions (as Euclidean polygons)
    i = 0
    centers = []
    for reg in regions:
        set = regions[reg]['region_set']
        centers.append(reg)
        for index in set:
            region = sv.regions[index]
            polygon = Poly3DCollection([sv.vertices[region]], alpha=1.0)
            polygon.set_color(colors[i])
            ax.add_collection3d(polygon)
        i += 1

    x_p, y_p, z_p = zip(*sv.points)
    x_p = [x_p[i] for i in centers]
    y_p = [y_p[i] for i in centers]
    z_p = [z_p[i] for i in centers]
    ax.scatter(x_p, y_p, z_p, c='k', s=1)
    plt.show()

def plot_points(points, losses):
    cart = [point['cartesian'] for point in points]
    x, y, z = zip(*cart)
    #fig = plt.figure()

    ma = max(losses)
    mi = min(losses)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    print(z)

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=losses, s=700)
    plt.show()

    #norm = matplotlib.colors.Normalize(vmin=mi, vmax=ma, clip=True)
    #mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greys_r)

    #loss_color = [mapper.to_rgba(l) for l in losses]


    #norm = matplotlib.colors.Normalize(mi, ma)
    #m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    #m.set_array([])
    #fcolors = m.to_rgba(losses)

    #X, Y = np.meshgrid(x, y)    # 50x50
    #Z = np.outer(z.T, z)        # 50x50

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_trisurf(x, y, z, rstride=1, cstride=1, color=losses, shade=0, cmap=cm.Greys_r)
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fcolors, shade=0)

def main():
    import faulthandler
    faulthandler.enable()
    #np.set_printoptions(threshold=sys.maxsize)
    path = sys.argv[1]
    points = np.load(os.path.join(path, 'points.npy'))
    losses = np.load(os.path.join(path, 'losses.npy'))
    print(losses)
    voronoi_plot(points, losses)


if __name__ == '__main__':
    main()
