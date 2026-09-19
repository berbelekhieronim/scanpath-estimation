<?php
/**
 * One-off Zenbox capability probe.
 *
 * Upload to the web root, open in a browser, note the results, then DELETE IT.
 * It reports server configuration and should not be left reachable.
 */
header('Content-Type: text/plain; charset=utf-8');

function yn($b) { return $b ? 'YES' : 'no'; }

$drivers = class_exists('PDO') ? PDO::getAvailableDrivers() : [];
$hasSqlite = in_array('sqlite', $drivers, true);
$hasMysql  = in_array('mysql', $drivers, true);

echo "SCANPATH DEMO — HOST CHECK\n";
echo str_repeat('=', 40) . "\n\n";

echo "PHP version:      " . PHP_VERSION . "\n";
echo "PHP >= 8.0:       " . yn(version_compare(PHP_VERSION, '8.0', '>=')) . "\n";
echo "PDO available:    " . yn(class_exists('PDO')) . "\n";
echo "PDO drivers:      " . ($drivers ? implode(', ', $drivers) : 'none') . "\n";
echo "  pdo_sqlite:     " . yn($hasSqlite) . "\n";
echo "  pdo_mysql:      " . yn($hasMysql) . "\n";
echo "GD (image size):  " . yn(extension_loaded('gd')) . "\n";
echo "JSON:             " . yn(extension_loaded('json')) . "\n\n";

echo "upload_max_filesize: " . ini_get('upload_max_filesize') . "\n";
echo "post_max_size:       " . ini_get('post_max_size') . "\n";
echo "memory_limit:        " . ini_get('memory_limit') . "\n\n";

// Can we write a data directory next to this file?
$dir = __DIR__ . '/_scanpath_write_test';
$canWrite = @mkdir($dir) && @file_put_contents($dir . '/t', 'x') !== false;
echo "Can create writable dir here: " . yn($canWrite) . "\n";
if ($canWrite) { @unlink($dir . '/t'); @rmdir($dir); }

// Can SQLite actually open a file database, not just claim the driver?
if ($hasSqlite) {
    try {
        $f = sys_get_temp_dir() . '/scanpath_probe_' . getmypid() . '.sqlite';
        $db = new PDO('sqlite:' . $f);
        $db->exec('CREATE TABLE t (a INTEGER)');
        $db->exec('INSERT INTO t VALUES (1)');
        $ok = $db->query('SELECT a FROM t')->fetchColumn() == 1;
        $db = null; @unlink($f);
        echo "SQLite file db read/write:   " . yn($ok) . "\n";
    } catch (Throwable $e) {
        echo "SQLite file db read/write:   FAILED — " . $e->getMessage() . "\n";
    }
}

echo "\n" . str_repeat('-', 40) . "\n";
if ($hasSqlite) {
    echo "VERDICT: use SqliteStorage. Nothing further needed.\n";
} elseif ($hasMysql) {
    echo "VERDICT: use MysqlStorage. Create a database in the Zenbox\n";
    echo "         control panel and note host/name/user/password.\n";
} else {
    echo "VERDICT: neither PDO driver present — contact Zenbox support,\n";
    echo "         or fall back to Option B hosting (spec section 4.1).\n";
}
echo "\n*** DELETE THIS FILE FROM THE SERVER NOW ***\n";
