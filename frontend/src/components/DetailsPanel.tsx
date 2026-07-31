// frontend/src/components/DetailsPanel.tsx

export default function DetailsPanel() {
  return (
    <aside
      style={{
        width: "300px",
        padding: "20px",
        background: "#0b1020",
        borderLeft: "1px solid rgba(255,255,255,0.1)",
        boxSizing: "border-box",
      }}
    >
      <h2
        style={{
          marginTop: 0,
          fontSize: "20px",
        }}
      >
        Details
      </h2>


      <div
        style={{
          marginTop: "30px",
          opacity: 0.5,
          fontSize: "14px",
        }}
      >
        Hover or select an item
        to display information here.
      </div>

    </aside>
  );
}