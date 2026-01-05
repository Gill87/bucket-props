"use client";

import { useEffect, useState } from "react";

type Pick = {
  player: string;
  line: number;
  predicted: number;
  pick: "OVER" | "UNDER";
  confidence: number;
  game_time?: string;
};

const formatGameTime = (iso?: string) => {
  if (!iso) return "";

  const date = new Date(iso);

  const datePart = date.toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
  });

  return `${datePart}`;
};

export default function Home() {
  const [picks, setPicks] = useState<Pick[]>([]);

  useEffect(() => {
    fetch("/picks.json")
      .then(res => res.json())
      .then(setPicks);
  }, []);

  return (
    <div style={styles.page}>
      {/* Header */}
      <header style={styles.header}>
        <div>
          <h1 style={styles.title}>BucketProps</h1>
          <p style={styles.subtitle}>Data-driven ML predictions</p>
        </div>
        <span style={styles.liveBadge}>● Live Picks</span>      
      </header>

      {/* Cards Grid */}
      <section style={styles.grid}>
          {[...picks]
            .sort((a, b) => b.confidence - a.confidence)
            .map((p, i) => {          
          const isOver = p.pick === "OVER";
          return (
            <div key={i} style={styles.card}>
              {/* Top row */}
              <div style={styles.cardTop}>
                <span style={styles.confidenceBadge(p.confidence)}>
                  {p.confidence}% Confidence
                </span>
              </div>

              {/* Player */}
              <h2 style={styles.player}>{p.player}</h2>

              {/* Date & Time*/}
              <p style={styles.gameTime}>{formatGameTime(p.game_time)}</p>

              {/* Line vs Projection */}
              <div style={styles.stats}>
                <div>
                  <p style={styles.statLabel}>Line</p>
                  <p style={styles.statValue}>{p.line}</p>
                </div>
                <div>
                  <p style={styles.statLabel}>Projection</p>
                  <p style={styles.projection}>{p.predicted}</p>
                </div>
              </div>

              {/* Pick button */}
              <div
                style={{
                  ...styles.pickButton,
                  background: isOver ? "#2ecc71" : "#e74c3c",
                }}
              >
                {isOver ? "↗ OVER" : "↘ UNDER"} {p.line}
              </div>
            </div>
          );
        })}
      </section>
    </div>
  );
}

/* ===============================
   Styles
================================ */

const styles: any = {
  page: {
    minHeight: "100vh",
    background: "linear-gradient(180deg, #0b1220, #05080f)",
    color: "#fff",
    padding: "2rem",
    fontFamily: "Inter, system-ui, sans-serif",
  },

  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "2rem",
  },

  title: {
    fontSize: "2rem",
    fontWeight: 700,
    margin: 0,
  },

  subtitle: {
    color: "#8fa3c8",
    marginTop: "0.25rem",
  },

  liveBadge: {
    background: "#0f2a1f",
    color: "#2ecc71",
    padding: "0.4rem 0.75rem",
    borderRadius: "999px",
    fontSize: "0.85rem",
  },

  gameTime: {
    fontSize: "0.85rem",
    color: "#8fa3c8",
    marginTop: "0.15rem",
  },

  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
    gap: "1.5rem",
  },

  card: {
    background: "linear-gradient(180deg, #121a2f, #0a1020)",
    borderRadius: "18px",
    padding: "1.5rem",
    boxShadow: "0 20px 40px rgba(0,0,0,0.35)",
    position: "relative",
  },

  cardTop: {
    display: "flex",
    justifyContent: "flex-end",
  },

  confidenceBadge: (conf: number) => ({
    background:
      conf >= 80 ? "#1f8f5f" : conf >= 65 ? "#1f4f8f" : "#2a2f45",
    color: "#fff",
    borderRadius: "999px",
    padding: "0.25rem 0.6rem",
    fontSize: "0.8rem",
  }),

  player: {
    marginTop: "0.75rem",
    fontSize: "1.2rem",
    fontWeight: 600,
  },

  stats: {
    display: "flex",
    justifyContent: "space-between",
    marginTop: "1.5rem",
    marginBottom: "1.5rem",
  },

  statLabel: {
    fontSize: "0.8rem",
    color: "#8fa3c8",
  },

  statValue: {
    fontSize: "1.6rem",
    fontWeight: 600,
  },

  projection: {
    fontSize: "1.6rem",
    fontWeight: 600,
    color: "#4da3ff",
  },

  pickButton: {
    marginTop: "auto",
    textAlign: "center" as const,
    padding: "0.9rem",
    borderRadius: "12px",
    fontWeight: 700,
    letterSpacing: "0.5px",
  },
};
