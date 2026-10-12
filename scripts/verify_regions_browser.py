"""Check rectangular live/replay geometry and legacy square views in Chromium."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


out=Path('runs/migration-analysis/browser');out.mkdir(parents=True,exist_ok=True)
errors=[];report={}
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1000})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8765/?compare=1&study=migration-10k-study')
    page.wait_for_function('window.__observatory?.renderCount>0')
    assert page.locator('#comparisoncard').is_visible()
    assert page.locator('#world').get_attribute('height')=='446'
    assert '左列' in page.locator('#worldtitle').inner_text()
    page.screenshot(path=str(out/'rectangular-comparison.png'),full_page=True)
    report['comparison_canvas']=page.evaluate('[canvas.width,canvas.height]')
    page.select_option('#layoutselect','single')
    page.select_option('#run','migration-L3-10k')
    page.wait_for_function('latest?.run==="migration-L3-10k" && mapArea!==null')
    assert page.locator('#world').get_attribute('height')=='225'
    report['single_canvas']=page.evaluate('[canvas.width,canvas.height]')
    assert page.evaluate('mapArea.scale===900/112')
    page.screenshot(path=str(out/'rectangular-live.png'),full_page=True)
    page.click('#replay');page.wait_for_function('clip?.frames_data.length>0',timeout=180000)
    clips=page.evaluate("()=>get('/api/replays?run=migration-L3-10k')")['clips']
    earliest=min(clips,key=lambda c:c['start'])
    page.select_option('#clipselect',earliest['id'])
    page.wait_for_function('(id)=>clip?.id===id',arg=earliest['id'])
    page.wait_for_function('clip?.config.width===112')
    assert page.locator('#world').get_attribute('height')=='225'
    frame=page.evaluate('current()')
    if frame['agents']:
        # Use the actual canvas pointer mapping, including its non-square height.
        a=frame['agents'][0];rect=page.locator('#world').bounding_box()
        page.mouse.click(rect['x']+(a['x']+.5)/112*rect['width'],rect['y']+(a['y']+.5)/28*rect['height'])
        assert page.locator('#details dd').count()==5
        report['clicked_agent']=a['id']
    page.screenshot(path=str(out/'rectangular-replay.png'),full_page=True)
    page.click('#play');page.wait_for_timeout(350);page.click('#play')
    assert int(page.locator('#scrub').input_value())>0
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
    page.screenshot(path=str(out/'rectangular-mobile.png'),full_page=True)
    page.set_viewport_size({'width':1440,'height':1000})
    page.goto('http://127.0.0.1:8765/?compare=1&study=half-food-10k-study')
    page.wait_for_function('window.__observatory?.renderCount>0')
    assert page.locator('#world').get_attribute('height')=='640'
    assert '上段' in page.locator('#worldtitle').inner_text()
    page.select_option('#layoutselect','single');page.select_option('#run','half-food-L3-10k')
    page.wait_for_function('latest?.run==="half-food-L3-10k" && mapArea!==null')
    assert page.locator('#world').get_attribute('height')=='900'
    page.goto('http://127.0.0.1:8765/?run=random-survival-metabolism-080-30seeds')
    page.wait_for_function('latest?.run==="random-survival-metabolism-080-30seeds" && window.__observatory.renderCount>0')
    assert page.locator('#run').input_value()=='random-survival-metabolism-080-30seeds'
    report['pinned_run_link']=True
    browser.close()
assert not errors,errors
report['javascript_errors']=errors
(out/'check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
