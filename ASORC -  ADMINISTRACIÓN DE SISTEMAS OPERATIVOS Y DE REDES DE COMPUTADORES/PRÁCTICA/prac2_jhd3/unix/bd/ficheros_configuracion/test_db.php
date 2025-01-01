<?php
$conn_string = "host=localhost port=5432 dbname=testdb user=testuser password=tupassword";
$conn = pg_connect($conn_string);

if (!$conn) {
    echo "Error de conexión.<br/>";
    echo "Error: " . pg_last_error();
    exit;
}

echo "<h1>Conexión exitosa a PostgreSQL</h1>";

$query = "SELECT * FROM usuarios";
$result = pg_query($conn, $query);

if (!$result) {
    echo "Error en la consulta.<br/>";
    echo "Error: " . pg_last_error();
    exit;
}

echo "<table border='1'>";
echo "<tr><th>ID</th><th>Nombre</th><th>Email</th></tr>";

while ($row = pg_fetch_assoc($result)) {
    echo "<tr>";
    echo "<td>" . htmlspecialchars($row['id']) . "</td>";
    echo "<td>" . htmlspecialchars($row['nombre']) . "</td>";
    echo "<td>" . htmlspecialchars($row['email']) . "</td>";
    echo "</tr>";
}

echo "</table>";

pg_close($conn);
?>
