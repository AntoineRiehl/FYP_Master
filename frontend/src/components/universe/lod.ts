//frontend/src/components/universe/lod.ts

export function getLOD(zoom: number) {

    if (zoom < 20) {
        return 0; // regions
    }

    if (zoom < 150) {
        return 1; // landmarks
    }

    return 2; // full atlas
}