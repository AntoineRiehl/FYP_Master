// frontend/src/components/sidebar/ColorModeSelector.tsx

import type {
    ColorMode
} from "../../config/colorMode";


// =====================================================
// PROPS
// =====================================================

type Props = {

    colorMode: ColorMode;

    onColorModeChange:
        (mode: ColorMode) => void;

};


// =====================================================
// COMPONENT
// =====================================================

export default function ColorModeSelector({

    colorMode,

    onColorModeChange

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
                Color by
            </h3>


            <select

                value={colorMode}

                onChange={(e) => {

                    onColorModeChange(
                        e.target.value as ColorMode
                    );

                }}

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

                <option value="domain">
                    Domain
                </option>

                <option value="cluster">
                    Cluster
                </option>

                <option value="category">
                    Category
                </option>

                <option value="feel">
                    Feel
                </option>

            </select>

        </section>

    );

}