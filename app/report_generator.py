def generate_report(responses, charts, output_file, logger):
    try:
        with open(output_file, "w") as f:
            f.write("# Exploratory Data Analysis Report\n\n")
            for response in responses:
                name = response['name']
                content = response['content'].replace("exitcode: 0 (execution succeeded)", " ")
                f.write(f"## {name.capitalize()}\n")
                f.write(content + "\n")
                for chart in charts:
                    if chart in content:
                        f.write(f"![{chart}](./charts/{chart})\n")
                f.write("\n\n")
        logger.info(f"Report saved to {output_file}")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise
