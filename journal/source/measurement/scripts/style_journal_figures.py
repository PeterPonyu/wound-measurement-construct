#!/usr/bin/env python3
"""Create and audit journal-only vector PDFs, leaving native figures immutable.

Requires the already-installed PyMuPDF, pikepdf, fontTools, Pillow and Inkscape (Tessera
schematics only). Run from any directory; paths are resolved from this script.
All text edits retain the original PDF drawing/image operators. Serif text runs
are fitted horizontally to their original advance, keeping the starting point,
baseline, rotation, point size, and the next text cursor position. Native math
font shows are copied verbatim. This is typography, never result regeneration.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unicodedata
import xml.etree.ElementTree as ET

import fitz
import pikepdf
from fontTools import subset
from PIL import Image, ImageDraw, ImageChops

FONT_DIR = Path('/home/zeyufu/.local/share/fonts/scmetaintel-article')
AUTHOR_RUNTIME = {
    'requested_model': 'gpt-6.1-sol', 'requested_effort': 'max',
    'runtime_reported_model': 'gpt-6.1-sol', 'runtime_reported_effort': 'max',
    'evidence': 'rollout-2026-10-11T20-14-20-01a12e1a-a4d3-7d82-be7b-5a8a76c61177.jsonl / turn_context',
    'scope': 'Authoring agent at initial creation, not a claim about later script reruns',
}
TEXT_OPS = {'BT','ET','Tf','Tz','Tc','Tw','Ts','TL','Tr','Tm','Td','TD','T*','Tj','TJ',"'",'"'}
SVG_NS = '{http://www.w3.org/2000/svg}'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(value):
    return unicodedata.normalize('NFKC', value).translate(str.maketrans({'−':'-', '–':'-', 'µ':'μ'}))


def compact(value):
    return ''.join(normalized(value).split())


def cmap(font):
    data = font.ToUnicode.read_bytes().decode('ascii')
    width = 2 if str(font.Subtype) == '/Type0' else 1
    result = {}
    for block in re.findall(r'beginbfchar(.*?)endbfchar', data, re.S):
        for src, dst in re.findall(r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', block):
            result[int(src, 16)] = bytes.fromhex(dst).decode('utf-16-be')
    for block in re.findall(r'beginbfrange(.*?)endbfrange', data, re.S):
        pattern = r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(<([0-9A-Fa-f]+)>|\[(.*?)\])'
        for m in re.finditer(pattern, block, re.S):
            start, end = int(m[1],16), int(m[2],16)
            if m[4]:
                dst = int(m[4],16)
                for code in range(start,end+1):
                    result[code] = (dst+code-start).to_bytes(len(m[4])//2,'big').decode('utf-16-be')
            else:
                for code, dst in zip(range(start,end+1), re.findall(r'<([0-9A-Fa-f]+)>',m[5])):
                    result[code] = bytes.fromhex(dst).decode('utf-16-be')
    return width, result


def codes(value, width):
    raw = bytes(value)
    assert len(raw) % width == 0
    return [int.from_bytes(raw[i:i+width],'big') for i in range(0,len(raw),width)]


def width_table(font):
    if str(font.Subtype) != '/Type0':
        return {int(font.FirstChar)+i: float(w) for i,w in enumerate(font.Widths)}, 1000.0
    descendant = font.DescendantFonts[0]
    values = list(descendant.get('/W', [])); result = {}; i = 0
    while i < len(values):
        start = int(values[i]); item = values[i+1]; i += 2
        if isinstance(item,pikepdf.Array):
            result.update({start+j:float(w) for j,w in enumerate(item)})
        else:
            end = int(item); w = float(values[i]); i += 1
            result.update({j:w for j in range(start,end+1)})
    return result,float(descendant.get('/DW',1000))


def instruction(op, *args):
    return (list(args),pikepdf.Operator(op))


def graphics_signature(page):
    def canonical(value):
        if isinstance(value,(int,float,Decimal)):
            return str(Decimal(str(value)).normalize())
        if isinstance(value,pikepdf.Array):
            return [canonical(v) for v in value]
        if isinstance(value,pikepdf.String):
            return {'bytes':bytes(value).hex()}
        return str(value)
    commands = [(str(i.operator),[canonical(v) for v in i.operands])
                for i in pikepdf.parse_content_stream(page) if str(i.operator) not in TEXT_OPS]
    return hashlib.sha256(json.dumps(commands,sort_keys=True).encode()).hexdigest()


def trace_chars(page, lower_panels=False, math_only=False):
    result = []
    for run in page.get_texttrace():
        text = ''.join(chr(c[0]) for c in run['chars'])
        if math_only and 'TeXGyreTermes' in run['font']:
            continue
        if lower_panels and 'bold' in run['font'].lower() and re.fullmatch('[A-H]',text):
            text = text.lower()
        result.append(text)
    return compact(''.join(result))


def images_signature(document):
    return sorted(sha_bytes(document.xref_stream(x)) for x in range(1,document.xref_length())
                  if document.xref_get_key(x,'Subtype')[1] == '/Image')


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def embedded_fonts(path):
    result = subprocess.run(['pdffonts',str(path)],check=True,text=True,capture_output=True).stdout
    lines = result.splitlines()[2:]
    return {'all_embedded':all(line.split()[-5]=='yes' for line in lines if line.strip()),
            'pdffonts':result}


def style_pdf(source, target, mode, fontdir):
    regular = fitz.Font(fontfile=str(fontdir/'Arial.ttf'))
    bold = fitz.Font(fontfile=str(fontdir/'Arial Bold.ttf'))
    with tempfile.TemporaryDirectory(prefix='journal-font-') as temp:
        intermediate = Path(temp)/'with-fonts.pdf'
        with fitz.open(source) as doc:
            characters={False:set(),True:set()}
            for page in doc:
                for run in page.get_texttrace():
                    value=''.join(chr(c[0]) for c in run['chars']);isbold='bold' in run['font'].lower()
                    if mode=='panels' and isbold and re.fullmatch('[A-H]',value):
                        characters[True].update(value.lower())
                    elif mode=='serif' and 'TeXGyreTermes' in run['font']:
                        characters[isbold].update(value)
            subsetpaths={}
            for isbold,chars in characters.items():
                if not chars:continue
                filename='Arial Bold.ttf' if isbold else 'Arial.ttf'
                options=subset.Options();options.retain_gids=True
                options.drop_tables+=['FFTM']
                font=subset.load_font(str(fontdir/filename),options)
                subsetter=subset.Subsetter(options=options);subsetter.populate(text=''.join(sorted(chars)))
                subsetter.subset(font);path=Path(temp)/filename;subset.save_font(font,str(path),options)
                subsetpaths[isbold]=path
            for page in doc:
                for isbold,path in subsetpaths.items():
                    page.insert_font(fontname='JournalArialBold' if isbold else 'JournalArial',fontfile=str(path))
            doc.save(intermediate,deflate=True)
        with pikepdf.open(intermediate) as doc:
            changes=[]; scales=[]; used_glyphs={'/JournalArial':{},'/JournalArialBold':{}}
            for page in doc.pages:
                instructions=pikepdf.parse_content_stream(page); output=[]
                fontname=None; fontsize=0; hscale=100; stack=[]
                metadata={}
                for name, font in page.Resources.Font.items():
                    if '/ToUnicode' in font:
                        width, mapping = cmap(font); widths, default = width_table(font)
                        metadata[str(name)] = (str(font.BaseFont),width,mapping,widths,default)
                for operands,operator in instructions:
                    op=str(operator)
                    if op=='q':stack.append((fontname,fontsize,hscale))
                    elif op=='Q':fontname,fontsize,hscale=stack.pop()
                    elif op=='Tf':fontname=str(operands[0]);fontsize=operands[1]
                    elif op=='Tz':hscale=float(operands[0])
                    if op not in {'TJ','Tj'} or fontname not in metadata:
                        output.append((operands,operator));continue
                    base,width,mapping,widths,default=metadata[fontname]
                    parts = list(operands[0]) if op=='TJ' else [operands[0]]
                    oldtext=''.join(mapping[c] for part in parts if isinstance(part,pikepdf.String) for c in codes(part,width))
                    panel = mode=='panels' and 'bold' in base.lower() and re.fullmatch('[A-H]',oldtext)
                    restyle = mode=='serif' and 'TeXGyreTermes' in base
                    if not (panel or restyle):
                        output.append((operands,operator));continue
                    usebold = 'bold' in base.lower(); targetfont=bold if usebold else regular
                    newfont = pikepdf.Name('/JournalArialBold' if usebold else '/JournalArial')
                    _,_,_,targetwidths,targetdefault=metadata[str(newfont)]
                    newparts=[]; oldtotal=0;newtotal=0
                    for part in parts:
                        if not isinstance(part,pikepdf.String):
                            newparts.append(part);oldtotal-=float(part);newtotal-=float(part);continue
                        glyphbytes=b''
                        for code in codes(part,width):
                            oldtotal+=widths.get(code,default)
                            text=mapping[code].lower() if panel else mapping[code]
                            for char in text:
                                gid=targetfont.has_glyph(ord(char))
                                assert gid or char=='\x00',(source.name,base,text,ord(char))
                                glyphbytes+=gid.to_bytes(2,'big')
                                newtotal+=targetwidths.get(gid,targetdefault)
                                previous=used_glyphs[str(newfont)].get(gid,char)
                                assert normalized(previous)==normalized(char),(gid,previous,char)
                                used_glyphs[str(newfont)][gid]=char
                        newparts.append(pikepdf.String(glyphbytes))
                    scale = 1.0 if panel else oldtotal/newtotal
                    assert .6 <= scale <= 1.4,(source.name,oldtext,scale)
                    output.extend([instruction('Tf',newfont,fontsize),
                                   instruction('Tz',Decimal(f'{hscale*scale:.9f}')),
                                   instruction('TJ',pikepdf.Array(newparts)),
                                   instruction('Tf',pikepdf.Name(fontname),fontsize),
                                   instruction('Tz',Decimal(str(hscale)))])
                    changes.append({'text':oldtext,'replacement':oldtext.lower() if panel else oldtext,
                                    'horizontal_scale':round(scale,6)})
                    scales.append(scale)
                page.Contents=doc.make_stream(pikepdf.unparse_content_stream(output))
            # Arial aliases hyphen and soft hyphen to one glyph. Write an explicit
            # used-glyph CMap so visible minus/hyphen characters stay extractable.
            for page in doc.pages:
                for name,glyphs in used_glyphs.items():
                    if not glyphs:continue
                    lines=[f'<{gid:04X}> <{char.encode("utf-16-be").hex().upper()}>' for gid,char in sorted(glyphs.items())]
                    data=('/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n'
                          '/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n'
                          '/CMapName /JournalArialUnicode def\n/CMapType 2 def\n'
                          '1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n')
                    for i in range(0,len(lines),100):
                        chunk=lines[i:i+100];data+=str(len(chunk))+' beginbfchar\n'+'\n'.join(chunk)+'\nendbfchar\n'
                    data+='endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n'
                    page.Resources.Font[name].ToUnicode=doc.make_stream(data.encode('ascii'))
            doc.save(target,deterministic_id=True)
    return {'method':'PDF text-show font replacement; original run advance preserved for serif runs',
            'changed_text_runs':len(changes),'panel_changes':changes if mode=='panels' else [],
            'horizontal_scale_range':[min(scales),max(scales)] if scales else [],
            'all_original_drawing_operations_preserved':True}


def style_svg(source, target, fontdir):
    tree=ET.parse(source);root=tree.getroot()
    source_texts=[e for e in root.iter() if e.tag==SVG_NS+'text']
    plain=[(''.join(e.itertext()),dict(e.attrib)) for e in source_texts]
    fontpaths={False:'/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf',
               True:'/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bold.otf'}
    original={k:fitz.Font(fontfile=v) for k,v in fontpaths.items()}
    arial={False:fitz.Font(fontfile=str(fontdir/'Arial.ttf')),True:fitz.Font(fontfile=str(fontdir/'Arial Bold.ttf'))}
    scales=[]
    for text in source_texts:
        value=''.join(text.itertext());bold=text.get('font-weight')=='bold'; x=float(text.get('x','0'))
        scale=original[bold].text_length(value)/arial[bold].text_length(value)
        text.set('font-family','Arial, sans-serif')
        assert 'transform' not in text.attrib
        text.set('transform',f'translate({x:g},0) scale({scale:.9f},1) translate({-x:g},0)')
        scales.append(scale)
    svg=target.with_suffix('.svg');tree.write(svg,encoding='utf-8',xml_declaration=True)
    subprocess.run(['inkscape',str(svg),'--export-type=pdf','--export-filename='+str(target)],
                   check=True,capture_output=True)
    reloaded=ET.parse(svg).getroot();after=[e for e in reloaded.iter() if e.tag==SVG_NS+'text']
    assert [v[0] for v in plain]==[''.join(e.itertext()) for e in after]
    # All non-text elements, attributes, coordinates and drawing order are identical.
    def nontext(node):
        return [(e.tag,sorted(e.attrib.items()),e.text or '') for e in node.iter() if e.tag!=SVG_NS+'text']
    assert nontext(ET.parse(source).getroot())==nontext(reloaded)
    return {'method':'Existing SVG re-export with Arial; horizontal fit preserves original text advance',
            'source_svg_sha256':sha(source),'derivative_svg_sha256':sha(svg),
            'all_source_svg_nontext_elements_unchanged':True,
            'all_source_svg_text_and_anchor_coordinates_unchanged':True,
            'changed_text_runs':len(plain),'horizontal_scale_range':[min(scales),max(scales)],
            'source_svg_text_tokens':plain}


def audit(source, target, mode, edit):
    unresolved=[]
    with fitz.open(source) as a,fitz.open(target) as b:
        assert len(a)==len(b)==1
        bounds=list(a[0].rect)==list(b[0].rect)
        before=trace_chars(a[0],lower_panels=mode=='panels');after=trace_chars(b[0])
        if mode=='svg':
            expected=compact(''.join(t[0] for t in edit['source_svg_text_tokens']))
            text_ok=expected==after
        else:text_ok=before==after
        before_numbers=Counter(re.findall(r'\d+(?:[.,]\d+)*',before))
        after_numbers=Counter(re.findall(r'\d+(?:[.,]\d+)*',after))
        if mode=='svg':before_numbers=Counter(re.findall(r'\d+(?:[.,]\d+)*',expected))
        numbers_ok=before_numbers==after_numbers
        images_ok=images_signature(a)==images_signature(b) if mode!='svg' else not a[0].get_images() and not b[0].get_images()
        with pikepdf.open(source) as original,pikepdf.open(target) as derivative:
            graphics_ok=graphics_signature(original.pages[0])==graphics_signature(derivative.pages[0]) if mode!='svg' else edit['all_source_svg_nontext_elements_unchanged']
        font_audit=embedded_fonts(target)
        native_math_before=[(s['font'],s['size'],s['dir'],s['chars']) for s in a[0].get_texttrace() if s['font'].startswith(('NewTX','tx','SFRM'))]
        native_math_after=[(s['font'],s['size'],s['dir'],s['chars']) for s in b[0].get_texttrace() if s['font'].startswith(('NewTX','tx','SFRM'))]
        # PDF real-number serialization can introduce sub-millipoint differences.
        def math_coords(runs):
            return [(r[0],r[1],r[2],c[0],c[1],c[2]) for r in runs for c in r[3]]
        old_math,new_math=math_coords(native_math_before),math_coords(native_math_after)
        math_delta=0.0;math_ok=len(old_math)==len(new_math)
        for old,new in zip(old_math,new_math):
            math_ok=math_ok and old[:5]==new[:5]
            math_delta=max(math_delta,abs(old[5][0]-new[5][0]),abs(old[5][1]-new[5][1]))
        math_ok=math_ok and math_delta < .001
        visible_fonts=sorted({s['font'] for s in b[0].get_texttrace()})
        sans_ok=all('Arial' in f or f.startswith(('NewTX','tx','SFRM')) for f in visible_fonts)
        outside=[];inherited_outside=[];partially_outside=[]
        for run in a[0].get_texttrace():
            box=fitz.Rect(run['bbox'])
            if not box.intersects(a[0].rect):
                inherited_outside.append({'text':''.join(chr(c[0]) for c in run['chars']),'bbox':list(box)})
        for run in b[0].get_texttrace():
            box=fitz.Rect(run['bbox'])
            if box.x0<-.1 or box.y0<-.1 or box.x1>b[0].rect.width+.1 or box.y1>b[0].rect.height+.1:
                outside.append({'text':''.join(chr(c[0]) for c in run['chars']),'bbox':list(box)})
                if box.intersects(b[0].rect):partially_outside.append(outside[-1])
        for label,ok in [('page bounds',bounds),('text tokens',text_ok),('number tokens',numbers_ok),('graphics',graphics_ok),('images',images_ok),('math symbols and coordinates',math_ok),('embedded fonts',font_audit['all_embedded']),('sans-serif lettering',sans_ok)]:
            if not ok:unresolved.append(label+' validation failed')
        if partially_outside:unresolved.append('Partly visible text trace bbox crosses page edge')
        if len(outside)-len(partially_outside)!=len(inherited_outside):unresolved.append('Changed fully off-page text count')
    return {'source_sha256':sha(source),'derivative_sha256':sha(target),
            'native_source_immutable':True,'page_bounds_preserved':bounds,
            'all_text_tokens_preserved_after_intended_normalization':text_ok,
            'all_number_tokens_preserved':numbers_ok,'all_graphics_preserved':graphics_ok,
            'all_image_streams_preserved':images_ok,'native_math_glyphs_and_origins_preserved':math_ok,
            'math_origin_maximum_delta_points':math_delta,'math_origin_tolerance_points':.001,
            'visible_fonts':visible_fonts,'sans_serif_lettering':sans_ok,
            'font_audit':font_audit,'inherited_fully_off_page_text_runs':inherited_outside,
            'off_page_text_runs':outside,'partially_clipped_text_runs':partially_outside,'unresolved':unresolved,
            'status':'PASS' if not unresolved else 'UNRESOLVED',**edit}


def render_sheet(source,target,qa,mode):
    name=source.stem
    images=[];full_images=[];rectangles=[]
    for label,path in [('native',source),('journal',target)]:
        with fitz.open(path) as doc:
            pix=doc[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
            out=qa/(name+'-'+label+'.png');pix.save(out)
            img=Image.open(out).convert('RGB');full_images.append(img.copy())
            for run in doc[0].get_texttrace():
                value=''.join(chr(c[0]) for c in run['chars'])
                changed=(mode=='panels' and 'bold' in run['font'].lower() and re.fullmatch('[A-Ha-h]',value)) or (mode=='serif' and ('TeXGyreTermes' in run['font'] or 'Arial' in run['font'])) or (mode=='svg' and label=='journal')
                if changed:
                    box=fitz.Rect(run['bbox'])+(-3,-3,3,3);rectangles.append(tuple(2*v for v in box))
            img.thumbnail((1000,1400));images.append(img)
    sheet=Image.new('RGB',(images[0].width+images[1].width+24,max(i.height for i in images)+40),'#dddddd')
    draw=ImageDraw.Draw(sheet);draw.text((8,8),'Native | Journal: '+name,fill='black')
    x=0
    for img in images:sheet.paste(img,(x,35));x+=img.width+24
    sheet.save(qa/(name+'-comparison.png'))
    difference=ImageChops.difference(*full_images).convert('L')
    mask=Image.new('L',difference.size,0);maskdraw=ImageDraw.Draw(mask)
    for box in rectangles:maskdraw.rectangle(box,fill=255)
    outside=ImageChops.multiply(difference,ImageChops.invert(mask))
    return {'resolution_dpi':144,'changed_pixels_total':sum(difference.histogram()[1:]),
            'changed_pixels_outside_lettering_regions':sum(outside.histogram()[1:]),
            'outside_difference_bbox':outside.getbbox(),
            'lettering_mask_padding_points':3,
            'scope':'Rendering comparison; source/replacement lettering bounding boxes plus padding masked; copy mode masks nothing'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,help='Candidate root; default is resolved from this script')
    parser.add_argument('--font-dir',type=Path,default=FONT_DIR)
    args=parser.parse_args();script=Path(__file__).resolve()
    root=args.root.resolve() if args.root else script.parents[2] if script.parents[1].name=='source' else script.parents[1]
    kind=root.name
    assert kind in {'measurement','population','tessera','raag'},kind
    manuscript=root/'source/manuscript' if kind=='tessera' else root
    source=manuscript/'figures';target=manuscript/'journal-figures';target.mkdir(exist_ok=True)
    qa=target/'qa';qa.mkdir(exist_ok=True)
    originals={p.name:sha(p) for p in source.glob('*.pdf')};records={}
    for pdf in sorted(source.glob('*.pdf')):
        output=target/pdf.name
        if kind=='raag':
            shutil.copy2(pdf,output);edit={'method':'Byte-for-byte copy of already-compliant native Arial/lowercase PDF','changed_text_runs':0};mode='copy'
        elif kind=='tessera' and pdf.with_suffix('.svg').exists():
            mode='svg';edit=style_svg(pdf.with_suffix('.svg'),output,args.font_dir)
        else:
            mode='serif' if kind=='tessera' else 'panels';edit=style_pdf(pdf,output,mode,args.font_dir)
        records[pdf.name]=audit(pdf,output,mode,edit)
        records[pdf.name]['raster_content_check']=render_sheet(pdf,output,qa,mode)
        if records[pdf.name]['raster_content_check']['changed_pixels_outside_lettering_regions']:
            records[pdf.name]['unresolved'].append('Rendered pixels changed outside lettering regions')
            records[pdf.name]['status']='UNRESOLVED'
        print(kind,pdf.name,records[pdf.name]['status'],flush=True)
    assert originals=={p.name:sha(p) for p in source.glob('*.pdf')},'A native PDF changed'
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'candidate':kind,
            'scope':'Journal-only typography derivatives; native figure/source PDFs immutable; no scientific recalculation',
            'author_runtime':AUTHOR_RUNTIME,'libraries':{'PyMuPDF':fitz.VersionBind,'pikepdf':pikepdf.__version__},
            'original_directory':str(source),'journal_directory':str(target),
            'rerun_command':'python '+str(script),'font_files':{n:sha(args.font_dir/n) for n in ['Arial.ttf','Arial Bold.ttf']},
            'normalization':'NFKC (including ligatures), whitespace, minus/en-dash typography and micro/mu; independent wound bold panel markers are intentionally lowercase',
            'figure_count':len(records),'figures':records,
            'visual_review':{'status':'PENDING','comparison_directory':str(qa),'scope':'Representative/dense figures require human/agent image inspection; script does not assert absence of collisions'},
            'status':'MECHANICAL_PASS_VISUAL_PENDING' if all(r['status']=='PASS' for r in records.values()) else 'UNRESOLVED'}
    (root/'journal-style-report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
