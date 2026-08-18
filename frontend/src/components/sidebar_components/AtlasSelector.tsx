// frontend/src/components/sidebar/AtlasSelector.tsx

import {
    ATLAS_REGISTRY
} from "../../config/atlasRegistry";


// =====================================================
// PROPS
// =====================================================

type Props = {

    atlasName: string;

    onAtlasChange:
        (name: string) => void;

};


// =====================================================
// COMPONENT
// =====================================================

export default function AtlasSelector({

    atlasName,

    onAtlasChange

}: Props) {

    return (

        <section
            style={{
                marginTop: "30px"
            }}
        >

            <h3
                style={{
                    fontSize: "14px",
                    opacity: 0.7
                }}
            >
                Atlas
            </h3>


            <select
                value={atlasName}

                onChange={(e) =>
                    onAtlasChange(
                        e.target.value
                    )
                }

                style={{
                    width: "100%",
                    padding: "8px",
                    background: "#111827",
                    color: "white",
                    border:
                        "1px solid rgba(255,255,255,0.1)",
                    borderRadius: "4px"
                }}
            >

                {
                    ATLAS_REGISTRY.map(
                        atlas => (

                            <option
                                key={atlas.id}
                                value={atlas.id}
                            >
                                {atlas.label}
                            </option>

                        )
                    )
                }

            </select>

        </section>

    );

}