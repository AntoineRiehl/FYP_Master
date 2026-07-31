//frontend/src/components/universe/lod.ts

export function getLOD(zoom: number) {

    if (zoom < 12) {
        return 0; // regions
    }

    if (zoom < 60) {
        return 1; // landmarks
    }

    return 2; // full atlas
}