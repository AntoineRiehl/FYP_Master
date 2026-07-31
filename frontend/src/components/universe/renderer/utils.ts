//frontend/src/components/universe/renderer/utils.ts

import type { Camera } from "../camera";
import { worldToScreen } from "../camera";


export function sampleArray<T>(
    arr:T[],
    max:number
){

    if(arr.length <= max)
        return arr;


    const step =
        Math.floor(arr.length / max);


    const out:T[] = [];


    for(
        let i = 0;
        i < arr.length;
        i += step
    ){

        out.push(arr[i]);

        if(out.length >= max)
            break;
    }


    return out;

}



export function distance(
    x1:number,
    y1:number,
    x2:number,
    y2:number
){

    return Math.hypot(
        x1-x2,
        y1-y2
    );

}



export function inView(
    camera:Camera,
    canvas:HTMLCanvasElement,
    x:number,
    y:number
){

    const p =
        worldToScreen(
            camera,
            canvas,
            x,
            y
        );


    return (
        p.x > -100 &&
        p.x < canvas.width + 100 &&
        p.y > -100 &&
        p.y < canvas.height + 100
    );

}