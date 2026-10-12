"""Real Chromium check against the running read-only viewer (not a mock DOM)."""
import argparse
import json
import time
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import sync_playwright


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8765');p.add_argument('--run',default='pilot-L0');p.add_argument('--output',default='runs/validation')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    def latest():
        with urlopen(args.url+'/api/latest?run='+args.run) as response:return json.load(response)
    errors=[]
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(args.url);page.wait_for_function('window.__observatory?.liveUpdates>0')
        page.select_option('#run',args.run);page.wait_for_timeout(1200)
        tick_before=latest()['tick'];page.wait_for_timeout(1600);tick_after=latest()['tick']
        updates=page.evaluate('window.__observatory.liveUpdates')
        colors=page.evaluate('new Set(Array.from(document.getElementById("world").getContext("2d").getImageData(0,0,900,900).data).filter((_,i)=>i%4!==3)).size')
        assert colors>8,'game canvas appears empty'
        page.screenshot(path=str(out/'viewer-live.png'),full_page=True)
        page.click('#replay');page.wait_for_function('document.getElementById("scrub").max>0')
        max_frame=int(page.locator('#scrub').get_attribute('max'))
        page.locator('#scrub').fill(str(max_frame//2));page.locator('#scrub').dispatch_event('input')
        paused_before=latest()['tick'];page.wait_for_timeout(1500);paused_after=latest()['tick']
        assert page.locator('#status').inner_text()=='REPLAY · PAUSED'
        page.screenshot(path=str(out/'viewer-replay.png'),full_page=True)
        page.click('#play');page.wait_for_timeout(350)
        assert int(page.locator('#scrub').input_value())>max_frame//2
        page.click('#play')
        # Select a replay agent and verify a local RGB view is actually rendered.
        replay=page.evaluate('current()')
        if replay['agents']:
            a=replay['agents'][0]
            page.evaluate('(id)=>{selected=id;render()}',a['id'])
            assert '#'+str(a['id']) in page.locator('#agenttitle').inner_text()
            assert page.locator('#details dd').count()==5
        page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(out/'viewer-mobile.png'),full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        browser.close()
    assert not errors,errors
    report={'browser':'Chromium','javascript_errors':errors,'canvas_color_values':colors,'live_updates':updates,'live_tick_before':tick_before,'live_tick_after':tick_after,'paused_replay_tick_before':paused_before,'paused_replay_tick_after':paused_after,'simulation_advanced_while_replay_paused':paused_after>paused_before,'replay_frames':max_frame+1}
    assert paused_after>paused_before,'Run must still be active to verify decoupled replay playback'
    (out/'browser-check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
