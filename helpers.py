# _*_ coding: utf-8 _*_
from lxml import etree
import time
import shutil

from BaseClass import *
from BaseModual import *
from DY_Line import *
import settings
import CommonElement
from CommonElement import Door
import main_bedroom.Base
import unknown.Base
import bedroom.Base
import numpy
import math

def is_inner_point(p,slist):
    #判断点是否在类似矩形的边界的图形内，在边界上返回 2 在界内返回 1 在界外返回 0
    m_len = 100000
    mm_len = -10000
    plist = [Segment2D(p,Point2D(p.x,mm_len)),Segment2D(p,Point2D(p.x,m_len)), \
             Segment2D(p, Point2D(p.y, mm_len)),Segment2D(p,Point2D(p.y,m_len))]
    ptag = [0,0,0,0]
    for s in slist:
        if s.seg.contains(p):
            return 2 #在边界上
        for i in range(4):
            if s.seg.intersection(plist[i])!=[]:
                ptag[i] += 1
    for item in ptag:
        if item==0 or item%2==0:
            return 0
    return 1


def point_distance(p1,p2):
    a= numpy.square(p1.x-p2.x) + numpy.square(p1.y-p2.y)
    a = math.sqrt(a)
    return a
def get_points_seg_intersect_boundary(seg, boundary, type=Line):
    """边界与某条线相交的所有点(去除线段）"""
    l = type(seg.p1, seg.p2)
    inter_pt = boundary.polygon.intersection(l)
    inter_pt = [pt for pt in inter_pt if isinstance(pt, Point2D)]
    return inter_pt
def get_points_seg_intersect_boundary_all(seg, boundary, type=Line):
    """边界与某条线相交的所有点(包含线段上的点）"""
    inter_pt_new = []
    xmin, ymin, xmax, ymax = boundary.polygon.bounds
    xlen = xmax - xmin
    ylen = ymax - ymin
    v = boundary.polygon.vertices
    l = type(seg.p1, seg.p2)
    inter_pt = boundary.polygon.intersection(l)
    if len(inter_pt):
        for in_pt in inter_pt:
            if isinstance(in_pt, Segment2D):
                inter_pt_new.append(in_pt.p1)
                inter_pt_new.append(in_pt.p2)
            elif isinstance(in_pt, Point2D):
                inter_pt_new.append(in_pt)
        for pt in inter_pt_new:
            if not seg.seg.contains(pt) and pt.x in [xmin, xmax] or pt.y in [ymin, ymax]:
                if pt in v:
                    return pt, True
                else:
                    return pt, False
        # return inter_pt_new
    else:
        return None

def get_intersect_line_from_boundary(seg,boundary):
    ''' 返回与seg有交点的所有边界'''
    intersect_line_list = []
    for l in boundary.seg_list:
        if l.seg.intersection(seg.seg)!=[]:
            intersect_line_list.append(l)
    return intersect_line_list

def get_adj_seg(ver,boundary):
    '''返回同一点的相邻边'''
    tp_adj_seg_list = []
    for seg in boundary.seg_list:
        if seg.seg.contains(ver):
            tp_adj_seg_list.append(seg)
    return tp_adj_seg_list
def another_p(line, point):
    '''返回一条边的另一个顶点'''
    if line.p1 == point:
        return line.p2
    else:
        return line.p1

def get_eles(ele_list, instance):
    """得到相应的element list"""
    targets = []
    for e in ele_list:
        if isinstance(e, instance):
            targets.append(e)
    return targets

def get_min_dis_seg_boundary(seg, boundary):
    """得到给定segment到boundary的最小距离，只计算bounary中与seg平行的线段"""
    para_line = get_paralleled_line(seg, boundary)
    dis = [l.distance(seg.p1) for l in para_line]
    return min(dis)

def get_paralleled_line(seg, boundary, type=Line):
    """得到与给定DY_segment平行的线段"""
    l_list = []
    for s in boundary.seg_list:
        if seg.line.is_parallel(s.line):
            l_list.append(type(s.p1, s.p2))
    return l_list

def get_ele_vertices_on_seg(ele, seg):
    """return组件在segment上的所有顶点"""
    vlist = []
    for v in ele.boundary.polygon.vertices:
        if seg.seg.contains(v):
            vlist.append(v)
    return vlist

def get_new_backline_with_bound(seg, bound):
    """通过boundary更新segment两点的顺序，使得seg一定沿着bound上的线段，以防反向"""
    v = bound.polygon.vertices
    assert seg.p1 in v, "backline端点不在边界上"
    assert seg.p2 in v, "backline端点不在边界上"

    for bs in bound.seg_list:
        if bs.seg.contains(seg.p1) and bs.seg.contains(seg.p2):
            if seg.normal.equals(bs.normal):
                return seg
            else:
                new_seg = DY_segment(seg.p2, seg.p1)
                return new_seg

def get_opposite_bounds(seg, boundary):
    op_list = []
    for s in boundary.seg_list:
        if seg.line.is_parallel(s.line) and seg.normal.p2.equals(s.normal.p2 * (-1)):
            op_list.append(s)
    return op_list

def get_adjacent_bounds(seg, boundary):
    adj_list = []
    for s in boundary.seg_list:
        if s.seg != seg.seg:
            if s.seg.contains(seg.p1) or s.seg.contains(seg.p2):
                adj_list.append(s)
    return adj_list

def get_adjacent_bounds_all(seg, boundary):
    adj_list = []
    for s in boundary.seg_list:
        if seg.line.is_perpendicular(s.line):
            adj_list.append(s)
    return adj_list

def get_out_door_name(in_door_name, door):
    for n in door.connect_list:
        if in_door_name is not n:
            out_door_name = n
    return out_door_name

def save_house_to_xml(house, file_name):
    root = etree.Element("XML")
    child = etree.SubElement(root, "House")
    child.set("name", house.name)
    for f in house.floor_list:
        child0 = etree.SubElement(child, "FloorPlan")
        child0.set("boundary", f.boundary.to_string())
        child0.set("name", f.name)
        for reg in f.region_list:
            child1 = etree.SubElement(child0, reg.__class__.__name__)
            child1.set("name", reg.name)
            child1.set("boundary", reg.boundary.to_string())
            child_temp = etree.SubElement(child1, 'Floor')
            child_temp.set("ID", str(reg.floor_id))
            child_temp = etree.SubElement(child1, 'SkirtingLine')
            child_temp.set("ID", str(reg.skirting_line_id))
            child_temp = etree.SubElement(child1, 'PlasterLine')
            child_temp.set("ID", str(reg.plaster_line_id))
            # child1_sub = etree.SubElement(child1, 'a')
            # child1_sub.text = ''
            for e in reg.ele_list:
                child2 = etree.SubElement(child1, e.__class__.__name__)
                child2.set("name", e.name)
                child2.set("boundary", e.boundary.to_string())
                child2.set("backline", e.backline.to_string())
                # child2.set("position", str(e.backline.p1).split('D')[1])
                child2.set("position", e.get_xyz_str())
                child2.set("angle", str(e.angle))
                child2.set("ID",str(e.ID))
                child2.set("back_len", str(int(e.backline.seg.length)))
                child2.set("front_len", str(int(e.len)))


                if e.__class__.__name__ == 'Door':
                    if e.door_body is not None:
                        child2.set("body", e.door_body.to_string())
                        child2.set("front_len", str(int(e.door_body.seg.length)))

                if e.is_multiple:
                    for ee in e.ele_list:
                        child3 = etree.SubElement(child2, ee.__class__.__name__)
                        child3.set("name", ee.name)
                        child3.set("boundary", ee.boundary.to_string())
                        child3.set("backline", ee.backline.to_string())
                        child3.set("position", ee.get_xyz_str())
                        # child3.set("position", str(ee.backline.p1).split('D')[1])
                        child3.set("angle", str(ee.angle))
            for l in reg.line_list:
                child2 = etree.SubElement(child1, l.__class__.__name__)
                child2.set("name", l.name)
                # child2.set("line", l.to_string())
                if hasattr(l, "boundary"):
                    child2.set("boundary", l.boundary.to_string())
                else:
                    child2.set("line", l.to_string())
                    child2.set("ID", str(l.ID))

    tree = etree.ElementTree(root)
    tree.write(file_name, pretty_print=True, xml_declaration=True, encoding='utf-8')
    return True

def is_boundary_intersection(b1, b2):
    """判断两个boundary是否相交，？"""
    vlist = b1.boundary.polygon.vertices

    for v in vlist:
        if b2.boundary.polygon.encloses_point(v):
            return True
    return False

def xml_set_boundary(key, region, node):
    """ 读取边界 """
    if key == DY_boundary.name:
        p_str_list = node.get(key).split(';')
        p_list = []
        for p_str in p_str_list:
            if p_str == '':
                continue
            list0 = p_str[1:-1].split(',')
            poi = Point(int(list0[0]), int(list0[1]))
            p_list.append(poi)
        eval_str = 'DY_boundary('
        for p in p_list:
            if p != p_list[-1]:
                eval_str += str(p) + ','
            else:
                eval_str += str(p)
        eval_str += ')'
        boundary = eval(eval_str)
        region.set_boundary(boundary)
        l_ver = len(boundary.polygon.vertices)
        '''此处是以前的分区调用，现在逻辑改到个区域里自己调用分区函数'''
        # if isinstance(region,main_bedroom.Base.MainBedroom) and l_ver >4:
        #         #or isinstance(region,bedroom.Base.Bedroom):
        #     #以列表形式返回
        #     t_list = get_virtual_boundary(boundary,node)
        #     virtual_boundary = t_list[0]
        #     region.set_virtual_boundary(virtual_boundary)
        #     #region 指的就是主卧这个区域
        #     #返回一个列表，列表第一项指示要进行虚拟区域布局
        #     t_list[0]=1
        #     return t_list
        #     pass
        # elif isinstance(region,unknown.Base.UnKnown):
        #
        #     temp_tuple1 = get_virtual_boundary1(boundary,node)
        #     virtual_boundary1 = temp_tuple1[0]
        #     if virtual_boundary1 is not None:
        #         if isinstance(region,Door) or isinstance(region,Window):
        #             pass
        #         else:
        #             region.set_virtual_boundary(virtual_boundary1)
        #         #region.set_boundary(virtual_boundary)
    else:
        pass


def get_virtual_boundary(origin_boundary,node):
    #这个函数作为单独的对主卧次卧进行处理的函数，不再考虑泛化，并返回针对的返回值
    #node1 的所有子节点列表，用于确定门墙啥的，
    inner_point = []
    xmin, ymin, xmax, ymax = origin_boundary.polygon.bounds
    xlen = xmax - xmin
    ylen = ymax - ymin
    for p in origin_boundary.polygon.vertices:
        if p.x != xmin and p.x != xmax and p.y != ymin and p.y != ymax:
            inner_point.append(p)
    verti = origin_boundary.polygon.vertices
    vertices_num = len(verti)
    in_poi_num = len(inner_point)    #定点数和内点数，用于区分阳台和刀把
    for v in verti:
        if (v in inner_point):
            verti.remove(v)
    # verti 外点，innerpoint内点

    # to find the position of the real door
    real_window = None #real window.    so does the window
    vc = None #virtual_curtain      if balcony exists in this area ,vc represent balcony
    vd = None #virtual_door    '''so does the door
    for no in node.getchildren():
        if no.tag == DY_Line.Window.__name__:
            real_window = no

        if no.tag == CommonElement.Door.__name__:
            vd = no
    if real_window != None:
        win_boundary = xml_get_boundary("boundary", real_window)
    if vd!=None:
        door_boundary = xml_get_boundary("boundary",vd)
        door_backline = xml_get_backline("backline",vd)

    #默认门不会贴着墙，即不会有交线段

    for te_l in get_adj_seg(door_backline.p1,door_boundary):
        if te_l.seg != door_backline.seg: #when it comes to compare two DY_segment object ,you have to use his attribute ,line or others
            te_ver_line0 = te_l
            break
    for te_l in get_adj_seg(door_backline.p2,door_boundary):
        if te_l.seg != door_backline.seg:
            te_ver_line1 = te_l
            break
    door_line_list = get_intersect_line_from_boundary(te_ver_line0,origin_boundary)
    door_line = door_line_list[0]
    vir_dr_p0 = te_ver_line0.seg.intersection(door_line.seg)
    vir_dr_p1 = te_ver_line1.seg.intersection(door_line.seg)
    vir_dr_mid_p = door_line.seg.midpoint

    '''门所在的墙和门墙上门的位置'''

    maxabcd = 0
    maxspoint = verti[0]
    for v in verti:
        a = v
        adj_seg = get_adj_seg(v,origin_boundary)
        ab = adj_seg[0]
        ac = adj_seg[1]
        if ab.seg.p1.x != ab.seg.p2.x:
            tem = ac
            ac = ab
            ab = tem
        if ab.seg.p1 == a:
            b = ab.seg.p2
        else:
            b = ab.seg.p1
        if ac.seg.p1 == a:
            c = ac.seg.p2
        else:
            c = ac.seg.p1
        if b in inner_point:
            bm = ab.seg.length
            bb = b
            tl0 = get_paralleled_line(ac, origin_boundary, DY_segment)
            for t0 in tl0:
                if t0.line.distance(a) > bm and ab.line.intersection(t0.seg) != []:  # 待会注意下焦点问题
                    bb = ab.line.intersection(t0.seg)[0]
                    bm = t0.line.distance(a)
            b = bb
            ab = DY_segment(a, b)
        if c in inner_point:
            cm = ac.seg.length
            cc = c
            tl1 = get_paralleled_line(ab, origin_boundary, DY_segment)
            for t1 in tl1:
                if t1.line.distance(a) > cm and ac.line.intersection(t1.seg) != []:  # 待会注意下焦点问题
                    cc = ac.line.intersection(t1.seg)[0]
                    cm = t1.line.distance(a)
            c = cc
            ac = DY_segment(a, c)


        #to initialize A,B,C ,then began to figure out whether there has cornor with vir_door or not.
        vi_tag = 'A' #use 'A,B,C' to indecate the position of the vir_door and real door 'A' MEANS NO
        Sabcd = (ab.seg.length) * (ac.seg.length)
        if Sabcd > maxabcd:
            maxabcd = Sabcd
            A = a
            B = b
            C = c
            AB = DY_segment(A, B)
            AC = DY_segment(A, C)
        else:
            continue
    # 去掉把
    D = C + AB.dir.p2 * (AB.seg.length)
    for v in verti:
        if (v in inner_point):
            verti.remove(v)
    # verti 外点，innerpoint内点

    if D in inner_point or D in verti:
        pass
    else:
        outtag = 1
        for s in origin_boundary.seg_list:
            if s.seg.contains(D):
                outtag = 0
                seg_cont_d = s
                break
        if outtag:
            # 不在边上
            yma = max(A.y, B.y, C.y, D.y)
            ymi = min(A.y, B.y, C.y, D.y)
            xma = max(A.x, B.x, C.x, D.x)
            xmi = min(A.x, B.x, C.x, D.x)
            dp = D
            smj = st = 0
            for vi in origin_boundary.polygon.vertices:
                if vi.x >= xmi and vi.x <= xma and vi.y >= ymi and vi.y <= yma:
                    st = abs(vi.x - A.x) * abs(vi.y - A.y)
                    if st > smj:
                        smj = st
                        dp = vi
                        # 得到阶梯最大值s 和点dp
            pb = B
            pc = C
            spb = spc = 0
            pdb = D
            pdc = D
            BD = DY_segment(B, D)
            CD = DY_segment(C, D)
            for s in origin_boundary.seg_list:
                if BD.seg.intersection(s.seg) != [] and s.line.is_parallel(AB.line):
                    pb = s.seg.intersection(BD.seg)[0]
                    if abs(pb.x - A.x) * abs(pb.y - A.y) > spb:
                        t1 = abs(pb.x - A.x)
                        t2 = abs(pb.y - A.y)
                        spb = abs(pb.x - A.x) * abs(pb.y - A.y)
                        pdb = pb
                if CD.seg.intersection(s.seg) != [] and s.line.is_parallel(AC.line):
                    pc = s.seg.intersection(CD.seg)[0]
                    if abs(pc.x - A.x) * abs(pc.y - A.y) > spc:
                        spc = abs(pc.x - A.x) * abs(pc.y - A.y)
                        pdc = pc
            if spb > spc:
                D = pdb
                C = A + AC.dir.p2 * (AB.seg.distance(D))
            else:
                D = pdc
                B = A + AB.dir.p2 * (AC.seg.distance(D))
            if smj < spb or smj < spc:
                pass
            else:
                D = dp
                B = Point2D(A.x, D.y)
                C = Point2D(D.x, A.y)

        else:
            print('!!!')
            D = Point2D(C.x, B.y)
            # 在边上，这种情况其实也不出现了

    AB = DY_segment(A, B)
    AC = DY_segment(A, C)
    BD = DY_segment(B, D)
    CD = DY_segment(C, D)
    finall_tag = 0
    yma = max(A.y, B.y, C.y, D.y)
    ymi = min(A.y, B.y, C.y, D.y)
    xma = max(A.x, B.x, C.x, D.x)
    xmi = min(A.x, B.x, C.x, D.x)

    inner_point_list = []
    for inp in inner_point:
        x = inp.x
        y = inp.y
        if x > xmi and x < xma and y > ymi and y < yma:
            inner_point_list.append(inp)
    back_tag = 0#回退标志
    for inp in inner_point_list:
        x = inp.x
        y = inp.y
        inner_list = []
        innerpo = prapo = inp
        horizontal_line = AC
        if x > xmi and x < xma and y > ymi and y < yma:
            inner_list = get_adj_seg(inp,origin_boundary)
            innum = 0
            len_inner_point_list = len(inner_point_list)
            for inline in inner_list:
                if (inline.seg.p1 in inner_point_list) and (inline.seg.p2 in inner_point_list):
                    innum += 1
                    vertical_line = inline
                    if inline.seg.p1 == inp:
                        innerpo = inline.seg.p2
                    else:
                        innerpo = inline.seg.p1
                    inner_point_list.remove(inline.seg.p1)
                    inner_point_list.remove(inline.seg.p2)

                else:
                    horizontal_line = inline
                    if inline.seg.p1 == inp:
                        innerpo = inline.seg.p2
                    else:
                        innerpo = inline.seg.p1
            if innum == 1:
                hvcomp0 = 0  # 指向CD边距
                hvcomp1 = 0  # 指向AB边距
                BD = DY_segment(B, D)
                CD = DY_segment(C, D)
                if AB.line.is_parallel(horizontal_line.line):
                    if AB.line.distance(inp) > AB.line.distance(innerpo):
                        if AB.line.distance(innerpo) > CD.line.distance(inp):
                            # 更新CD
                            hvcomp0 = 1
                            hvcomp1 = AB.line.distance(innerpo)
                        else:
                            # 更新AB
                            hvcomp0 = CD.line.distance(inp)
                            hvcomp1 = 1
                    else:
                        if AB.line.distance(inp) > CD.line.distance(innerpo):
                            # 更新CD
                            hvcomp0 = 1
                            hvcomp1 = AB.line.distance(inp)
                        else:
                            # 更新AB
                            hvcomp0 = CD.line.distance(innerpo)
                            hvcomp1 = 1
                    if len_inner_point_list == 2 or len_inner_point_list == 1:
                        # 等于1这里暂时存疑
                        if AC.line.distance(inp) > BD.line.distance(inp):
                            # 更新BD
                            dis = AC.line.distance(inp)
                            B = A + AB.dir.p2 * dis
                            D = C + AB.dir.p2 * dis
                            back_tag = 1
                            new_line = DY_segment(B,D)
                        else:
                            dis = BD.line.distance(inp)
                            A = B - AB.dir.p2 * dis
                            C = D - AB.dir.p2 * dis
                            back_tag = 1
                            new_line = DY_segment(A, C)
                    else:
                        if hvcomp0 == 1:
                            C = A + AC.dir.p2 * hvcomp1
                            D = B + AC.dir.p2 * hvcomp1
                            back_tag = 1
                            new_line = DY_segment(C, D)
                        elif hvcomp1 == 1:
                            A = C - AC.dir.p2 * hvcomp0
                            B = D - AC.dir.p2 * hvcomp0
                            back_tag = 1
                            new_line = DY_segment(A,B)
                elif AC.line.is_parallel(horizontal_line.line):
                    if AC.line.distance(inp) > AC.line.distance(innerpo):
                        if AC.line.distance(innerpo) > BD.line.distance(inp):
                            # 更新BD
                            hvcomp0 = 1
                            hvcomp1 = AC.line.distance(innerpo)
                        else:
                            # 更新AC
                            hvcomp0 = BD.line.distance(inp)
                            hvcomp1 = 1
                    else:
                        if AC.line.distance(inp) > BD.line.distance(innerpo):
                            # 更新BD
                            hvcomp0 = 1
                            hvcomp1 = AC.line.distance(inp)
                        else:
                            # 更新AC
                            hvcomp0 = BD.line.distance(innerpo)
                            hvcomp1 = 1
                    if len_inner_point_list == 2 or len_inner_point_list == 1:
                        if AB.line.distance(inp) > CD.line.distance(inp):
                            dis = AB.line.distance(inp)
                            C = A + AC.dir.p2 * dis
                            D = B + AC.dir.p2 * dis
                            back_tag = 1
                            new_line = DY_segment(C, D)
                        else:
                            dis = CD.line.distance(inp)
                            A = C - AC.dir.p2 * dis
                            B = D - AC.dir.p2 * dis
                            back_tag = 1
                            new_line = DY_segment(A, B)
                    else:
                        if hvcomp0 == 1:
                            B = A + AB.dir.p2 * hvcomp1
                            D = C + AB.dir.p2 * hvcomp1
                            back_tag = 1
                            new_line = DY_segment(B, D)
                        elif hvcomp1 == 1:
                            A = B - AB.dir.p2 * hvcomp0
                            C = D - AB.dir.p2 * hvcomp0
                            back_tag = 1
                            new_line = DY_segment(A, C)
                yma = max(A.y, B.y, C.y, D.y)
                ymi = min(A.y, B.y, C.y, D.y)
                xma = max(A.x, B.x, C.x, D.x)
                xmi = min(A.x, B.x, C.x, D.x)
            else:
                continue

        '''the code above had ensure the position of A,B,C,D.
        '''
    virtual_boundary = DY_boundary(*[A, C, D, B])
    v_d_tag = 1
    for s in virtual_boundary.seg_list:
        if s.seg.contains(vir_dr_mid_p):
            door_line = s
            v_d_tag = 0
            break
            #door 的其他参数跟初始没变。有p1 p2 mid
    # para_dr_line = []
    # vertical_dr_line = []
    # para_dr_line = get_paralleled_line(door_line, virtual_boundary, DY_segment)
    # vertical_dr_line = get_paralleled_line(te_ver_line1,virtual_boundary,DY_segment)
    vir_ver_list = []
    if v_d_tag:
        for p in virtual_boundary.polygon.vertices:
            long = point_distance(p,vir_dr_mid_p)
            vir_ver_list.append([p,long])
        vir_ver_list.sort(key=lambda x:x[1],reverse=False)
        #不在需要门的位置了，用顶点距离来表征门等的相对位置。最近点就是门附近，布局从远点开始就行，也可以不用care镜像
        door_line = Segment2D(vir_ver_list[0][0],vir_ver_list[1][0])
        #降序排列，最近的一个定为A点。     假设门要真的出现AB距离相等的话，可以强制锁X,Y，遇到再说
        #这里doorline的定义基于他是把结构，如果是长鞋结构会有问题，暂时不管了反正是区域

    else:
        for p in virtual_boundary.polygon.vertices:
            long = point_distance(p, vir_dr_mid_p)
            vir_ver_list.append([p, long])
        vir_ver_list.sort(key=lambda x: x[1])
        # 降序排列，最近的一个定个点就是A点。
    balcony_line = None
    b_tag = 0
    inner_point_num = len(inner_point)
    if vertices_num >= 6 and inner_point_num>=2:
        b_tag = 1
        balcony_line = Segment2D(vir_ver_list[2][0],vir_ver_list[3][0])
    elif back_tag:
        b_tag = 1
        balcony_line = new_line
    else:
        pass
    temp_list = [virtual_boundary,vir_ver_list,[v_d_tag,door_line,vir_dr_mid_p],[b_tag,balcony_line]]
    #llist形式返回所需要的数据
    return temp_list #假如门在阳台上，再说吧

def get_virtual_boundary1(origin_boundary):
    inner_point = []
    xmin, ymin, xmax, ymax = origin_boundary.polygon.bounds
    xlen = xmax - xmin
    ylen = ymax - ymin
    for p in origin_boundary.polygon.vertices:
        if p.x != xmin and p.x != xmax and p.y != ymin and p.y != ymax:
            inner_point.append(p)
    verti = origin_boundary.polygon.vertices
    for v in verti:
        if (v in inner_point):
            verti.remove(v)
    # verti 外点，innerpoint内点
    maxabcd = 0
    maxspoint = verti[0]
    for v in verti:
        a = v
        adj_seg = get_adj_seg(v,origin_boundary)
        ab = adj_seg[0]
        ac = adj_seg[1]
        if ab.seg.p1.x != ab.seg.p2.x:
            tem = ac
            ac = ab
            ab = tem
        if ab.seg.p1 == a:
            b = ab.seg.p2
        else:
            b = ab.seg.p1
        if ac.seg.p1 == a:
            c = ac.seg.p2
        else:
            c = ac.seg.p1
        if b in inner_point:
            bm = ab.seg.length
            bb = b
            tl0 = get_paralleled_line(ac, origin_boundary, DY_segment)
            for t0 in tl0:
                if t0.line.distance(a) > bm and ab.line.intersection(t0.seg) != []:  # 待会注意下焦点问题
                    bb = ab.line.intersection(t0.seg)[0]
                    bm = t0.line.distance(a)
            b = bb
            ab = DY_segment(a, b)
        if c in inner_point:
            cm = ac.seg.length
            cc = c
            tl1 = get_paralleled_line(ab, origin_boundary, DY_segment)
            for t1 in tl1:
                if t1.line.distance(a) > cm and ac.line.intersection(t1.seg) != []:  # 待会注意下焦点问题
                    cc = ac.line.intersection(t1.seg)[0]
                    cm = t1.line.distance(a)
            c = cc
            ac = DY_segment(a, c)

        Sabcd = (ab.seg.length) * (ac.seg.length)
        if Sabcd > maxabcd:
            maxabcd = Sabcd
            A = a
            B = b
            C = c
            AB = DY_segment(A, B)
            AC = DY_segment(A, C)
        else:
            continue
    # 去掉把
    D = C + AB.dir.p2 * (AB.seg.length)
    for v in verti:
        if (v in inner_point):
            verti.remove(v)
    # verti 外点，innerpoint内点

    if D in inner_point or D in verti:
        pass
    else:
        outtag = 1
        for s in origin_boundary.seg_list:
            if s.seg.contains(D):
                outtag = 0
                seg_cont_d = s
                break
        if outtag:
            # 不在边上
            yma = max(A.y, B.y, C.y, D.y)
            ymi = min(A.y, B.y, C.y, D.y)
            xma = max(A.x, B.x, C.x, D.x)
            xmi = min(A.x, B.x, C.x, D.x)
            dp = D
            smj = st = 0
            for vi in origin_boundary.polygon.vertices:
                if vi.x >= xmi and vi.x <= xma and vi.y >= ymi and vi.y <= yma:
                    st = abs(vi.x - A.x) * abs(vi.y - A.y)
                    if st > smj:
                        smj = st
                        dp = vi
                        # 得到阶梯最大值s 和点dp
            pb = B
            pc = C
            spb = spc = 0
            pdb = D
            pdc = D
            BD = DY_segment(B, D)
            CD = DY_segment(C, D)
            for s in origin_boundary.seg_list:
                if BD.seg.intersection(s.seg) != [] and s.line.is_parallel(AB.line):
                    pb = s.seg.intersection(BD.seg)[0]
                    if abs(pb.x - A.x) * abs(pb.y - A.y) > spb:
                        t1 = abs(pb.x - A.x)
                        t2 = abs(pb.y - A.y)
                        spb = abs(pb.x - A.x) * abs(pb.y - A.y)
                        pdb = pb
                if CD.seg.intersection(s.seg) != [] and s.line.is_parallel(AC.line):
                    pc = s.seg.intersection(CD.seg)[0]
                    if abs(pc.x - A.x) * abs(pc.y - A.y) > spc:
                        spc = abs(pc.x - A.x) * abs(pc.y - A.y)
                        pdc = pc
            if spb > spc:
                D = pdb
                C = A + AC.dir.p2 * (AB.seg.distance(D))
            else:
                D = pdc
                B = A + AB.dir.p2 * (AC.seg.distance(D))
            if smj < spb or smj < spc:
                pass
            else:
                D = dp
                B = Point2D(A.x, D.y)
                C = Point2D(D.x, A.y)

        else:
            print('!!!')
            D = Point2D(C.x, B.y)
            # 在边上，这种情况其实也不出现了

    AB = DY_segment(A, B)
    AC = DY_segment(A, C)
    BD = DY_segment(B, D)
    CD = DY_segment(C, D)
    yma = max(A.y, B.y, C.y, D.y)
    ymi = min(A.y, B.y, C.y, D.y)
    xma = max(A.x, B.x, C.x, D.x)
    xmi = min(A.x, B.x, C.x, D.x)

    inner_point_list = []
    for inp in inner_point:
        x = inp.x
        y = inp.y
        if x > xmi and x < xma and y > ymi and y < yma:
            inner_point_list.append(inp)
    for inp in inner_point_list:
        x = inp.x
        y = inp.y
        inner_list = []
        innerpo = prapo = inp
        horizontal_line = AC
        if x > xmi and x < xma and y > ymi and y < yma:
            inner_list = get_adj_seg(inp,origin_boundary)
            innum = 0
            len_inner_point_list = len(inner_point_list)
            for inline in inner_list:
                if (inline.seg.p1 in inner_point_list) and (inline.seg.p2 in inner_point_list):
                    innum += 1
                    vertical_line = inline
                    if inline.seg.p1 == inp:
                        innerpo = inline.seg.p2
                    else:
                        innerpo = inline.seg.p1
                    inner_point_list.remove(inline.seg.p1)
                    inner_point_list.remove(inline.seg.p2)

                else:
                    horizontal_line = inline
                    if inline.seg.p1 == inp:
                        innerpo = inline.seg.p2
                    else:
                        innerpo = inline.seg.p1
            if innum == 1:
                hvcomp0 = 0  # 指向CD边距
                hvcomp1 = 0  # 指向AB边距
                BD = DY_segment(B, D)
                CD = DY_segment(C, D)
                if AB.line.is_parallel(horizontal_line.line):
                    if AB.line.distance(inp) > AB.line.distance(innerpo):
                        if AB.line.distance(innerpo) > CD.line.distance(inp):
                            # 更新CD
                            hvcomp0 = 1
                            hvcomp1 = AB.line.distance(innerpo)
                        else:
                            # 更新AB
                            hvcomp0 = CD.line.distance(inp)
                            hvcomp1 = 1
                    else:
                        if AB.line.distance(inp) > CD.line.distance(innerpo):
                            # 更新CD
                            hvcomp0 = 1
                            hvcomp1 = AB.line.distance(inp)
                        else:
                            # 更新AB
                            hvcomp0 = CD.line.distance(innerpo)
                            hvcomp1 = 1
                    if len_inner_point_list == 2 or len_inner_point_list == 1:
                        # 等于1这里暂时存疑
                        if AC.line.distance(inp) > BD.line.distance(inp):
                            # 更新BD
                            dis = AC.line.distance(inp)
                            B = A + AB.dir.p2 * dis
                            D = C + AB.dir.p2 * dis
                        else:
                            dis = BD.line.distance(inp)
                            A = B - AB.dir.p2 * dis
                            C = D - AB.dir.p2 * dis
                    else:
                        if hvcomp0 == 1:
                            C = A + AC.dir.p2 * hvcomp1
                            D = B + AC.dir.p2 * hvcomp1
                        elif hvcomp1 == 1:
                            A = C - AC.dir.p2 * hvcomp0
                            B = D - AC.dir.p2 * hvcomp0
                elif AC.line.is_parallel(horizontal_line.line):
                    if AC.line.distance(inp) > AC.line.distance(innerpo):
                        if AC.line.distance(innerpo) > BD.line.distance(inp):
                            # 更新BD
                            hvcomp0 = 1
                            hvcomp1 = AC.line.distance(innerpo)
                        else:
                            # 更新AC
                            hvcomp0 = BD.line.distance(inp)
                            hvcomp1 = 1
                    else:
                        if AC.line.distance(inp) > BD.line.distance(innerpo):
                            # 更新BD
                            hvcomp0 = 1
                            hvcomp1 = AC.line.distance(inp)
                        else:
                            # 更新AC
                            hvcomp0 = BD.line.distance(innerpo)
                            hvcomp1 = 1
                    if len_inner_point_list == 2 or len_inner_point_list == 1:
                        if AB.line.distance(inp) > CD.line.distance(inp):
                            dis = AB.line.distance(inp)
                            C = A + AC.dir.p2 * dis
                            D = B + AC.dir.p2 * dis
                        else:
                            dis = CD.line.distance(inp)
                            A = C - AC.dir.p2 * dis
                            B = D - AC.dir.p2 * dis
                    else:
                        if hvcomp0 == 1:
                            B = A + AB.dir.p2 * hvcomp1
                            D = C + AB.dir.p2 * hvcomp1
                        elif hvcomp1 == 1:
                            A = B - AB.dir.p2 * hvcomp0
                            C = D - AB.dir.p2 * hvcomp0
                yma = max(A.y, B.y, C.y, D.y)
                ymi = min(A.y, B.y, C.y, D.y)
                xma = max(A.x, B.x, C.x, D.x)
                xmi = min(A.x, B.x, C.x, D.x)
            else:
                continue

    virtual_boundary = DY_boundary(*[A, C, D, B])


    return virtual_boundary

class VirtualBoundary(object):

    def __init__(self, *args, **kwargs):
        self.has_diagonal = False
        self.diagonal_seg = []
        self.vertices_OF = False

        polygon_tmp = Polygon(*args, **kwargs)
        self.polygon = Polygon(*polygon_tmp.vertices)

        self.check_the_boundary()
        self.virtual_polygon = []

        self.seg_list = []
        self.virtual_seg_list = []
        self.remove_hilt()
        self.__set_seg_list()

        self.if_virtual = False

    def remove_hilt(self):
        inner_point = []
        xmin, ymin, xmax, ymax = self.polygon.bounds
        xlen = xmax - xmin
        ylen = ymax - ymin
        for p in self.polygon.vertices:
            if p.x != xmin and p.x != xmax and p.y != ymin and p.y != ymax:
                inner_point.append(p)
        v = self.polygon.vertices
        vertices_temp = []
        succ = False
        if len(inner_point):
            for i in range(-len(v), 0):
                seg_temp = DY_segment(v[i], v[i + 1])
                self.seg_list.append(seg_temp)
                for in_pt in inner_point:
                    if seg_temp.seg.contains(in_pt) and \
                            (seg_temp.seg.length >= (xlen // 2) or seg_temp.seg.length >= (ylen // 2)):
                        inter_pt = get_points_seg_intersect_boundary(seg_temp, self)[0]
                        # self.seg_list.remove(seg_temp)
                        # if v[i] == in_pt:
                        #     seg_temp = DY_segment(v[i + 1], inter_pt)
                        # elif v[i + 1] == in_pt:
                        #     seg_temp = DY_segment(v[i], inter_pt)
                        # self.seg_list.append(seg_temp)
                        succ = True
                        for p in v:
                            if p.x == in_pt.x or p.y == in_pt.y:
                                if not seg_temp.seg.contains(p):
                                    idx = v.index(p)
                                    v[idx] = in_pt
                            if p.x == inter_pt.x or p.x == inter_pt.y:
                                inter_pt_seg = DY_segment(inter_pt, p)
                                if inter_pt_seg.dir.p2 != seg_temp.normal.p2:
                                    idx = v.index(p)
                                    v[idx] = inter_pt
                        self.polygon = Polygon(*v)
                        break
                # if succ:
                #     break


    def __set_seg_list(self):
        self.seg_list.clear()
        v = self.polygon.vertices
        if not self.vertices_OF and  self.has_diagonal is False:
            self.if_virtual = False
            self.virtual_polygon = None #means stay with self.polygon
            for i in range(-len(v), 0):
                self.seg_list.append(DY_segment(v[i], v[i + 1]))

        elif self.vertices_OF and self.has_diagonal is False:
            self.if_virtual = True
            if len(v) < 4:
                raise Exception("error:不支持此户型")
            v = self.polygon.vertices
            for i in range(-len(v), 0):
                self.seg_list.append(DY_segment(v[i], v[i + 1]))
            op_seg = {}
            # seg_alone = False
            for seg in self.seg_list:

                adj_seg = get_adjacent_bounds(seg, self)
                seg_dis_max = sorted(adj_seg, key=lambda s: s.seg.length)[-1]
                op_seg[seg] = get_opposite_bounds(seg, self)
                dis = sorted(op_seg[seg], key=lambda l: l.line.distance(seg.p1))
                delta = dis[-1].seg.length - dis[0].seg.length
                if abs(delta) > (seg_dis_max.seg.length // 2):
                    seg_alone = False
                elif delta == 0:
                    seg_alone = False
                else:
                    seg_alone = True
                # to build the rect
                #first we should remove the outside rect we call it hilt

                if seg_alone:
                    for op in op_seg[seg]:
                        if op.line.distance(seg.p1) == dis[0].line.distance(seg.p1):
                            found_seg = op
                            intersect_p = []
                            for adj in adj_seg:
                                intersect_p.append(found_seg.line.intersection(adj.line)[0])
                            #here we should make the Polygon has no intersecting sides
                            for pt in intersect_p:
                                if pt.x == seg.p2.x or pt.y == seg.p2.y:
                                    p3 = pt
                                else:
                                    p4 = pt
                            self.virtual_polygon = Polygon(*[seg.p1, seg.p2, p3, p4])
                            v_vir = self.virtual_polygon.vertices
                            for i in range(-len(v_vir), 0):
                                self.virtual_seg_list.append(DY_segment(v_vir[i], v_vir[i + 1]))
                            # break
                # break
        elif not self.vertices_OF and self.has_diagonal:
            self.if_virtual = True
        #TODO only diagonal  exists

        elif self.vertices_OF and  self.has_diagonal:
            self.if_virtual = True
        #TODO points overflow and diagonal

        else:
            raise Exception("warning:当前不支持此户型（圆弧？）")


    def check_the_boundary(self):

        v = self.polygon.vertices
        if len(v) != 4:
            self.vertices_OF = True
        for i in range(-len(v), 0):
            if v[i].x == v[i + 1].x or v[i].y == v[i + 1].y:
                continue
            else:
                self.has_diagonal = True
                self.diagonal_seg.append(Segment2D(v[i], v[i + 1]))

    def to_string(self):

        res = ''
        for i in self.seg_list:
            res += point_to_string(i.p1) + ';'
        return res[:-1]

    def draw(self, ax, ls='-', col='#000000'):
        for s in self.seg_list:
            s.draw(ax, ls, col)
        for s in self.virtual_seg_list:
            s.draw(ax, '--', '#990033')

    def draw_virtual(self, ax, ls= '--', col='#990033'):
        for s in self.virtual_seg_list:
            s.draw(ax, ls, col)












# def xml_set_window(key, region, node):
    # p_str_list = node.get(key).split(';')
    # p_list = []
    # for p_str in p_str_list:
    #     list0 = p_str[1:-1].split(',')
    #     poi = Point(int(list0[0]), int(list0[1]))
    #     p_list.append(poi)
    # eval_str = 'DY_Line.Window('
    # for p in p_list:
    #     if p != p_list[-1]:
    #         eval_str += str(p) + ','
    #     else:
    #         eval_str += str(p)
    # eval_str += ')'
    # win = eval(eval_str)
    # region.add_window(win)

def xml_get_boundary(key, node):
    """ 读取边界 """
    if key == DY_boundary.name:
        p_str_list = node.get(key).split(';')
        p_list = []
        for p_str in p_str_list:
            if p_str == '':
                continue
            list0 = p_str[1:-1].split(',')
            poi = Point(int(list0[0]), int(list0[1]))
            p_list.append(poi)
        eval_str = 'DY_boundary('
        for p in p_list:
            if p != p_list[-1]:
                eval_str += str(p) + ','
            else:
                eval_str += str(p)
        eval_str += ')'
        boundary = eval(eval_str)
        return boundary
    else:
        return None


def xml_set_window(region, node):
    boundary = xml_get_boundary("boundary", node)
    coincide_seg = boundary.polygon.intersection(region.boundary.polygon)
    coincide_seg = [seg for seg in coincide_seg if isinstance(seg, Segment2D)]

    for seg in coincide_seg:
        window = DY_Line.Window(seg.p1, seg.p2)
        window.set_boundary(boundary)
        region.add_window(window)

def xml_set_border(key, region, node):
    p_str_list = node.get(key).split(';')
    p_list = []
    for p_str in p_str_list:
        list0 = p_str[1:-1].split(',')
        poi = Point(int(list0[0]), int(list0[1]))
        p_list.append(poi)
    eval_str = 'DY_Line.Border('
    for p in p_list:
        if p != p_list[-1]:
            eval_str += str(p) + ','
        else:
            eval_str += str(p)
    eval_str += ')'
    bord = eval(eval_str)
    region.add_border(bord)

def xml_set_backline(key, ele, node):
    if key == "backline":
        p_str_list = node.get(key).split(';')
        p_list = []
        for p_str in p_str_list:
            list0 = p_str[1:-1].split(',')
            poi = Point(int(list0[0]), int(list0[1]))
            p_list.append(poi)

        assert len(p_list)==2, "backline 只能有两个点"
        backline = DY_segment(p_list[0], p_list[1])
        ele.set_backline(backline)
    else:
        pass

def xml_get_backline(key,node):
    if key == "backline":
        p_str_list = node.get(key).split(';')
        p_list = []
        for p_str in p_str_list:
            list0 = p_str[1:-1].split(',')
            poi = Point(int(list0[0]), int(list0[1]))
            p_list.append(poi)

        assert len(p_list)==2, "backline 只能有两个点"
        backline = DY_segment(p_list[0], p_list[1])
        return backline
    else:
        pass

def xml_set_door_body(key, door, node):
    if key == "body" and node.get(key) != None:
        p_str_list = node.get(key).split(';')
        p_list = []
        for p_str in p_str_list:
            list0 = p_str[1:-1].split(',')
            poi = Point(int(list0[0]), int(list0[1]))
            p_list.append(poi)

        assert len(p_list)==2, "body 只能有两个点"
        body = DY_segment(p_list[0], p_list[1])
        door.set_body(body)
    else:
        pass

def xml_set_door(door, node):
    att = 'attribute'
    # if node.get(att) is None:
    #     door.set_type(settings.DOOR_TYPE[0])
    # else:
    #     if node.get(att) == settings.DOOR_TYPE[0]:
    #         door.set_type(settings.DOOR_TYPE[0])
    #     elif node.get(att) == settings.DOOR_TYPE[1]:
    #         door.set_type(settings.DOOR_TYPE[1])
    #     elif node.get(att) == settings.DOOR_TYPE[2]:
    #         door.set_type(settings.DOOR_TYPE[2])
    #
    # if node.get(att) != settings.DOOR_TYPE[2]:
    #     xml_set_boundary("boundary", door, node)
    #     xml_set_door_body("body", door, node)
    # xml_set_backline("backline", door, node)

    if node.get(att) is None:
        door.set_type(settings.DOOR_TYPE[0])
    xml_set_boundary("boundary", door, node)
    xml_set_door_body("body", door, node)
    xml_set_backline("backline", door, node)

import os
def listfile(dirname, postfix = ''):
    filelist = []
    files = os.listdir(dirname)
    dirname = dirname + '/'
    for item in files:
        #filelist.append([dirname,item])
        if os.path.isfile(dirname+item):
            if item.endswith(postfix):
                filelist.append(item)
        else:
            if os.path.isdir(dirname+item):
                pass
                # filelist.extend(listfile(dirname+item+'/',postfix))
    return filelist

def error_log(log, time_str):
    error_log_check()

    xml = etree.parse('error.xml')
    root = xml.getroot()
    error_str = str(log).split(':')
    if error_str[0] == 'error' or error_str[0] == 'warning':
        child = etree.SubElement(root, error_str[0])
        child.set("time", time_str)
        child.text = error_str[1]
    else:
        child = etree.SubElement(root, "error")
        child.set("time", time_str)
        child.text = str(log)

    tree = etree.ElementTree(root)
    tree.write('error.xml', pretty_print=True, xml_declaration=True, encoding='utf-8')

def get_error_replica(fname, time_str):
    if os.path.exists('error_xml') is False:
        os.mkdir('error_xml')
    src = fname
    dst = 'error_xml//' + os.path.basename(fname)[:-4] + '_' + time_str + '.xml'
    shutil.copy(src, dst)

def error_log_check():
    error_file = 'error.xml'
    if os.path.exists(error_file):
        os.remove(error_file)
    root = etree.Element('ERRORANDWARNING')
    tree = etree.ElementTree(root)
    tree.write(error_file, pretty_print=True, xml_declaration=True, encoding='utf-8')

