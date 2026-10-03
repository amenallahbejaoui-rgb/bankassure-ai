import { NextResponse } from "next/server";

const BACKEND_URL =
  process.env.BACKEND_URL || "http://127.0.0.1:8000";

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/transactions`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return NextResponse.json(
        {
          error: "Failed to fetch transactions from BankAssure API",
          status: response.status,
        },
        { status: response.status }
      );
    }

    const transactions = await response.json();

    return NextResponse.json(transactions);
  } catch (error) {
    console.error("Transaction proxy error:", error);

    return NextResponse.json(
      {
        error: "BankAssure backend is unavailable",
      },
      { status: 503 }
    );
  }
}