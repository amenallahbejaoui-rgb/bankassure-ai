"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

type Decision = "APPROVE" | "REVIEW" | "REJECT";

type Transaction = {
  id: number;
  transaction_number: string;
  account_id: number;
  amount: string;
  currency: string;
  transaction_type: string;
  timestamp: string;
  merchant: string;
  merchant_category: string;
  country: string;
  is_international: boolean;
  merchant_risk: string;
  is_fraud: boolean;
  status: string;
  fraud_score: number | null;
  fraud_decision: Decision | null;
  fraud_reasons: string[] | null;
};

export default function Home() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const loadTransactions = useCallback(async () => {
    try {
      setError("");

      const response = await fetch("/api/transactions", {
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error("Failed to load transactions");
      }

      const data = await response.json();

      if (!Array.isArray(data)) {
        throw new Error("Invalid transaction response");
      }

      setTransactions(data);
      setLastUpdated(new Date());
    } catch (err) {
      console.error(err);
      setError(
        "Unable to connect to the BankAssure backend. Make sure FastAPI is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    queueMicrotask(loadTransactions);

    const interval = setInterval(loadTransactions, 10000);

    return () => clearInterval(interval);
  }, [loadTransactions]);

  const summary = useMemo(() => {
    const scored = transactions.filter(
      (transaction) => transaction.fraud_decision
    );

    const approve = scored.filter(
      (transaction) => transaction.fraud_decision === "APPROVE"
    ).length;

    const review = scored.filter(
      (transaction) => transaction.fraud_decision === "REVIEW"
    ).length;

    const reject = scored.filter(
      (transaction) => transaction.fraud_decision === "REJECT"
    ).length;

    const fraud = transactions.filter(
      (transaction) => transaction.is_fraud
    ).length;

    const averageScore =
      scored.length > 0
        ? scored.reduce(
            (sum, transaction) =>
              sum + (transaction.fraud_score ?? 0),
            0
          ) / scored.length
        : 0;

    return {
      total: transactions.length,
      scored: scored.length,
      approve,
      review,
      reject,
      fraud,
      averageScore,
    };
  }, [transactions]);

  const recentTransactions = useMemo(() => {
    return [...transactions]
      .sort(
        (a, b) =>
          new Date(b.timestamp).getTime() -
          new Date(a.timestamp).getTime()
      )
      .slice(0, 10);
  }, [transactions]);

  return (
    <main className="min-h-screen bg-[#07111f] text-white">
      {/* Header */}
      <header className="border-b border-white/10 bg-[#091525]/95">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between px-6 py-5">
          <div>
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-500/15 text-blue-400">
                <ShieldIcon />
              </div>

              <div>
                <h1 className="text-xl font-semibold tracking-tight">
                  BankAssure AI
                </h1>

                <p className="text-xs text-slate-400">
                  Banking & Insurance Intelligence Platform
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden text-right sm:block">
              <p className="text-xs text-slate-500">
                Model 1
              </p>

              <p className="text-sm font-medium text-slate-200">
                Fraud Detection
              </p>
            </div>

            <div className="flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1.5">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              <span className="text-xs font-medium text-emerald-300">
                Live
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-[1500px] px-6 py-8">
        {/* Page heading */}
        <section className="mb-8">
          <div className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
            <div>
              <p className="mb-2 text-sm font-medium text-blue-400">
                FRAUD INTELLIGENCE
              </p>

              <h2 className="text-3xl font-semibold tracking-tight">
                Fraud Detection Dashboard
              </h2>

              <p className="mt-2 max-w-2xl text-sm text-slate-400">
                Real-time visibility into transaction risk, fraud
                decisions, and model activity.
              </p>
            </div>

            <div className="text-sm text-slate-500">
              {lastUpdated
                ? `Updated ${lastUpdated.toLocaleTimeString()}`
                : "Connecting..."}
            </div>
          </div>
        </section>

        {/* Error */}
        {error && (
          <div className="mb-6 rounded-2xl border border-red-400/20 bg-red-400/10 p-4 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* KPI cards */}
        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
          <MetricCard
            label="Total Transactions"
            value={summary.total.toLocaleString()}
            description="Transactions loaded from API"
            icon={<ActivityIcon />}
          />

          <MetricCard
            label="Scored"
            value={summary.scored.toLocaleString()}
            description="Transactions evaluated by fraud model"
            icon={<BrainIcon />}
          />

          <MetricCard
            label="Rejected"
            value={summary.reject.toLocaleString()}
            description="High-risk transactions"
            icon={<AlertIcon />}
            emphasis="danger"
          />

          <MetricCard
            label="Under Review"
            value={summary.review.toLocaleString()}
            description="Transactions requiring review"
            icon={<EyeIcon />}
            emphasis="warning"
          />

          <MetricCard
            label="Avg. Fraud Score"
            value={`${(summary.averageScore * 100).toFixed(1)}%`}
            description="Average scored transaction"
            icon={<ChartIcon />}
          />
        </section>

        {/* Decision overview */}
        <section className="mt-6 grid gap-6 lg:grid-cols-3">
          <div className="rounded-2xl border border-white/10 bg-[#0b192b] p-6 lg:col-span-2">
            <div className="mb-6 flex items-center justify-between">
              <div>
                <h3 className="font-semibold">
                  Fraud Decision Overview
                </h3>

                <p className="mt-1 text-sm text-slate-500">
                  Current model decisions
                </p>
              </div>

              <div className="rounded-lg bg-white/5 px-3 py-2 text-xs text-slate-400">
                Thresholds: 0.35 / 0.75
              </div>
            </div>

            <div className="space-y-5">
              <DecisionBar
                label="APPROVE"
                count={summary.approve}
                total={summary.scored}
                className="bg-emerald-400"
                textClass="text-emerald-300"
              />

              <DecisionBar
                label="REVIEW"
                count={summary.review}
                total={summary.scored}
                className="bg-amber-400"
                textClass="text-amber-300"
              />

              <DecisionBar
                label="REJECT"
                count={summary.reject}
                total={summary.scored}
                className="bg-red-400"
                textClass="text-red-300"
              />
            </div>
          </div>

          <div className="rounded-2xl border border-white/10 bg-[#0b192b] p-6">
            <h3 className="font-semibold">
              Model Status
            </h3>

            <p className="mt-1 text-sm text-slate-500">
              Fraud ensemble
            </p>

            <div className="mt-6 space-y-4">
              <ModelStatus name="Logistic Regression" />
              <ModelStatus name="XGBoost" />
              <ModelStatus name="Isolation Forest" />
              <ModelStatus name="Ensemble Decision Engine" />
            </div>
          </div>
        </section>

        {/* Recent transactions */}
        <section className="mt-6 rounded-2xl border border-white/10 bg-[#0b192b]">
          <div className="flex flex-col justify-between gap-3 border-b border-white/10 p-6 sm:flex-row sm:items-center">
            <div>
              <h3 className="font-semibold">
                Recent Transactions
              </h3>

              <p className="mt-1 text-sm text-slate-500">
                Live transaction activity from BankAssure
              </p>
            </div>

            <button
              onClick={loadTransactions}
              className="rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-300 transition hover:bg-white/10"
            >
              Refresh
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[900px] text-left text-sm">
              <thead className="border-b border-white/10 text-xs uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="px-6 py-4">Transaction</th>
                  <th className="px-6 py-4">Merchant</th>
                  <th className="px-6 py-4">Amount</th>
                  <th className="px-6 py-4">Risk</th>
                  <th className="px-6 py-4">Score</th>
                  <th className="px-6 py-4">Decision</th>
                  <th className="px-6 py-4">Status</th>
                </tr>
              </thead>

              <tbody className="divide-y divide-white/5">
                {loading ? (
                  <tr>
                    <td
                      colSpan={7}
                      className="px-6 py-12 text-center text-slate-500"
                    >
                      Loading transaction intelligence...
                    </td>
                  </tr>
                ) : recentTransactions.length === 0 ? (
                  <tr>
                    <td
                      colSpan={7}
                      className="px-6 py-12 text-center text-slate-500"
                    >
                      No transactions found.
                    </td>
                  </tr>
                ) : (
                  recentTransactions.map((transaction) => (
                    <TransactionRow
                      key={transaction.id}
                      transaction={transaction}
                    />
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>

        {/* Footer */}
        <footer className="py-8 text-center text-xs text-slate-600">
          BankAssure AI · Model 1 Fraud Detection · Live backend data
        </footer>
      </div>
    </main>
  );
}

function MetricCard({
  label,
  value,
  description,
  icon,
  emphasis,
}: {
  label: string;
  value: string;
  description: string;
  icon: React.ReactNode;
  emphasis?: "danger" | "warning";
}) {
  const valueClass =
    emphasis === "danger"
      ? "text-red-300"
      : emphasis === "warning"
        ? "text-amber-300"
        : "text-white";

  return (
    <div className="rounded-2xl border border-white/10 bg-[#0b192b] p-5">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
            {label}
          </p>

          <p className={`mt-3 text-3xl font-semibold ${valueClass}`}>
            {value}
          </p>
        </div>

        <div className="rounded-xl bg-white/5 p-2.5 text-slate-400">
          {icon}
        </div>
      </div>

      <p className="mt-3 text-xs text-slate-500">
        {description}
      </p>
    </div>
  );
}

function DecisionBar({
  label,
  count,
  total,
  className,
  textClass,
}: {
  label: string;
  count: number;
  total: number;
  className: string;
  textClass: string;
}) {
  const percentage =
    total > 0 ? Math.max((count / total) * 100, count > 0 ? 1 : 0) : 0;

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span className={`text-sm font-medium ${textClass}`}>
          {label}
        </span>

        <span className="text-sm text-slate-400">
          {count.toLocaleString()}
        </span>
      </div>

      <div className="h-2 overflow-hidden rounded-full bg-white/5">
        <div
          className={`h-full rounded-full ${className} transition-all duration-500`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

function ModelStatus({ name }: { name: string }) {
  return (
    <div className="flex items-center justify-between rounded-xl border border-white/5 bg-white/[0.02] px-4 py-3">
      <span className="text-sm text-slate-300">{name}</span>

      <span className="flex items-center gap-2 text-xs text-emerald-300">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
        Active
      </span>
    </div>
  );
}

function TransactionRow({
  transaction,
}: {
  transaction: Transaction;
}) {
  const score = transaction.fraud_score ?? 0;

  const decisionClass =
    transaction.fraud_decision === "REJECT"
      ? "border-red-400/20 bg-red-400/10 text-red-300"
      : transaction.fraud_decision === "REVIEW"
        ? "border-amber-400/20 bg-amber-400/10 text-amber-300"
        : transaction.fraud_decision === "APPROVE"
          ? "border-emerald-400/20 bg-emerald-400/10 text-emerald-300"
          : "border-white/10 bg-white/5 text-slate-400";

  return (
    <tr className="transition hover:bg-white/[0.025]">
      <td className="px-6 py-4">
        <div className="font-medium text-slate-200">
          {transaction.transaction_number}
        </div>

        <div className="mt-1 text-xs text-slate-500">
          #{transaction.id}
        </div>
      </td>

      <td className="px-6 py-4">
        <div className="text-slate-300">
          {transaction.merchant}
        </div>

        <div className="mt-1 text-xs capitalize text-slate-500">
          {transaction.merchant_category}
        </div>
      </td>

      <td className="px-6 py-4 font-medium text-slate-200">
        {Number(transaction.amount).toLocaleString()}{" "}
        {transaction.currency}
      </td>

      <td className="px-6 py-4">
        <span
          className={
            Number(transaction.merchant_risk) >= 50
              ? "text-red-300"
              : Number(transaction.merchant_risk) >= 20
                ? "text-amber-300"
                : "text-slate-400"
          }
        >
          {Number(transaction.merchant_risk).toFixed(1)}
        </span>
      </td>

      <td className="px-6 py-4">
        <span className="font-medium text-slate-200">
          {(score * 100).toFixed(1)}%
        </span>
      </td>

      <td className="px-6 py-4">
        <span
          className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-medium ${decisionClass}`}
        >
          {transaction.fraud_decision ?? "UNSCORED"}
        </span>
      </td>

      <td className="px-6 py-4">
        <span className="text-xs capitalize text-slate-400">
          {transaction.status}
        </span>
      </td>
    </tr>
  );
}

function ShieldIcon() {
  return (
    <svg
      width="21"
      height="21"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
    >
      <path d="M12 3 20 6v5c0 5-3.4 8.7-8 10-4.6-1.3-8-5-8-10V6l8-3Z" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  );
}

function ActivityIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M3 12h4l3-8 4 16 3-8h4" />
    </svg>
  );
}

function BrainIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M9 4a3 3 0 0 0-3 3v1a3 3 0 0 0-2 5 3 3 0 0 0 3 5h2" />
      <path d="M15 4a3 3 0 0 1 3 3v1a3 3 0 0 1 2 5 3 3 0 0 1-3 5h-2" />
      <path d="M9 4v16M15 4v16M9 9h6M9 15h6" />
    </svg>
  );
}

function AlertIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M12 3 22 20H2L12 3Z" />
      <path d="M12 9v5M12 17h.01" />
    </svg>
  );
}

function EyeIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function ChartIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M4 19V5M4 19h16" />
      <path d="m7 15 4-5 3 3 5-7" />
    </svg>
  );
}