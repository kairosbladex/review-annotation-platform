"""Experimental crop sheet only; original source and locator are unchanged."""
import math
from PIL import Image, ImageDraw

def project_box(box, matrix, source_size, target_size):
    """Preserve unclipped coordinates; reject any clipping for original evidence."""
    if len(box)!=4 or not all(math.isfinite(v) for v in box) or not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):
        raise ValueError('invalid screenshot box')
    sw,sh=source_size;tw,th=target_size
    if min(sw,sh,tw,th)<=0:raise ValueError('invalid image size')
    points=[]
    for x,y in ((box[0],box[1]),(box[2],box[1]),(box[0],box[3]),(box[2],box[3])):
        q=[sum(a*b for a,b in zip(row,(x*sw,y*sh,1))) for row in matrix]
        if len(q)!=3 or not all(math.isfinite(v) for v in q) or abs(q[2])<1e-12:raise ValueError('invalid projection')
        points.append((q[0]/q[2],q[1]/q[2]))
    raw=[min(x for x,y in points),min(y for x,y in points),max(x for x,y in points),max(y for x,y in points)]
    clipped=[max(0,raw[0]),max(0,raw[1]),min(tw,raw[2]),min(th,raw[3])]
    area=(raw[2]-raw[0])*(raw[3]-raw[1])
    if area<=0:raise ValueError('degenerate projection')
    retained=max(0,clipped[2]-clipped[0])*max(0,clipped[3]-clipped[1])/area
    return {'unclipped_px':raw,'clipped_px':clipped,'retained_fraction':retained,'complete':all(abs(a-b)<1e-8 for a,b in zip(raw,clipped)),'bbox':[v/[tw,th][i%2] for i,v in enumerate(clipped)]}

def make_sheet(source,candidates,layout):
    if layout not in ('legacy','context'):raise ValueError('unknown layout')
    height=280 if layout=='legacy' else 560
    sheet=Image.new('RGB',(1280,((len(candidates)+4)//5)*height),'white');draw=ImageDraw.Draw(sheet)
    for i,c in enumerate(candidates):
        b=c['bbox'];x1,y1,x2,y2=b[0]*source.width,b[1]*source.height,b[2]*source.width,b[3]*source.height
        mx,my=max(20,(x2-x1)*.25),max(20,(y2-y1)*.25)
        crop=source.crop((max(0,x1-mx),max(0,y1-my),min(source.width,x2+mx),min(source.height,y2+my)))
        crop.thumbnail((256,256));px,py=(i%5)*256,(i//5)*height
        sheet.paste(crop,(px,py+24));draw.text((px,py),str(i),fill='black')
        if layout=='context':
            # Same target, wider physical context, no in-image text or generated pixels.
            cx,cy=(x1+x2)/2,(y1+y2)/2;side=max(320,(x2-x1)*6,(y2-y1)*6)
            context=source.crop((max(0,cx-side/2),max(0,cy-side/2),min(source.width,cx+side/2),min(source.height,cy+side/2)))
            context.thumbnail((256,256));sheet.paste(context,(px,py+304));draw.text((px,py+280),str(i)+' context',fill='black')
    return sheet
