export default function Message({ role, content }) {
  const cls = role === "user" ? "message user" : "message assistant";
  return (
    <div className={cls}>
      <p>{content}</p>
    </div>
  );
}
